import os
import shutil
import ast

import logging
import time
import numpy as np
import torch
from torch import nn
from enum import Enum, auto
from pathlib import Path

import faiss
import faiss.contrib.torch_utils

logger = logging.getLogger(__name__)
logger.setLevel(20)

class DIST(Enum):
    l2 = auto()
    dot = auto()

    @staticmethod
    def from_string(s):
        try:
            return DIST[s.lower()]
        except KeyError:
            raise ValueError()

class KEY_TYPE(Enum):
    last_ffn_input = auto()
    last_ffn_output = auto()

    @staticmethod
    def from_string(s):
        try:
            return KEY_TYPE[s.lower()]
        except KeyError:
            raise ValueError()

class KNNWrapper(object):  
    def __init__(self, dstore_sizes, dstore_dir, dimension, 
            knn_sim_func=None, knn_keytype=None,
            no_load_keys=False, move_dstore_to_mem=False, knn_gpu=True,
            recompute_dists = False,
            k=16, lmbdas="[[0.25]]", knn_temp=1.0, probe=32, on_the_fly=False, test_references=(), corrections="",
            dstore_num=1):    
        self.dstore_sizes = ast.literal_eval(dstore_sizes)
        self.dstore_dir = dstore_dir
        self.dimension = dimension
        self.dstore_num = dstore_num

        self.k = k
        self.knn_temperature = knn_temp
        self.probe = probe
        self.knn_sim_func = DIST.l2 if knn_sim_func is None else knn_sim_func
        self.knn_keytype = KEY_TYPE.last_ffn_input if knn_keytype is None else knn_keytype
        self.no_load_keys = no_load_keys
        self.recompute_dists = recompute_dists
        self.move_dstore_to_mem = move_dstore_to_mem
        self.knn_gpu = knn_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 0
       
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lmbdas = torch.tensor(ast.literal_eval(lmbdas)).to(self.device)
        self.prompt_input_ids = None
        self.keys = None
        self.values = None
        self.prompt_attention_mask = None
        self.model = None
        self.vocab_size = None
        self.activation_capturer = None
        self.is_encoder_decoder = None
        self.hook_handles = []
        

        dist_type_to_dist_func = {
            DIST.l2: KNNWrapper.l2,
            DIST.dot: KNNWrapper.dotprod,
        }
        self.dist_func = dist_type_to_dist_func[knn_sim_func] # l2 or dot product function
        #references to use for on-the-fly human feedback
        self.on_the_fly = on_the_fly
        if self.on_the_fly:
            self.test_references = test_references
            self.ref_counter = 0
            self.dstore_idx = self.dstore_sizes[0]
            self.corrections = corrections
            #If using gpu, a cpu index is needed because it's not possible to call add_with_ids for
            #index in gpu as of this moment: https://github.com/facebookresearch/faiss/pull/2263
            #This means tat on-the-fly can't be used with indexes in gpu as of yet
            self.knn_gpu = False

    def setup_faiss(self, dstore_idx):
        if not self.dstore_dir:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()
        index_name = get_index_path(self.dstore_dir, self.model.config.model_type, self.dstore_sizes[dstore_idx-1], self.dimension, dstore_idx) 
        cpu_index = faiss.read_index(index_name, faiss.IO_FLAG_ONDISK_SAME_DIR)
        logger.info(f'Reading datastore took {time.time() - start} s')
        cpu_index.nprobe = self.probe

        if self.knn_gpu:
            start = time.time()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index, co)
            logger.info(f'Moving index to GPU took {time.time() - start} s')
        else:
            gpu_index = cpu_index

        # make_direct_map() allows calling reconstruct(n), 
        # and reconstructing key vectors given their ids
        # currently, this is implemented only for CPU indexes:
        # https://github.com/facebookresearch/faiss/issues/2181
        #cpu_index.make_direct_map()

        keys_vals_prefix = get_dstore_path(self.dstore_dir, self.model.config.model_type, self.dstore_sizes[dstore_idx-1], self.dimension, dstore_idx)

        if not self.no_load_keys:
            keys = np.memmap(f'{keys_vals_prefix}_keys.npy', dtype=np.float16, mode='r',
                                  shape=(self.dstore_sizes[dstore_idx-1], self.dimension))
        else: 
            keys = None
        vals = np.memmap(f'{keys_vals_prefix}_vals.npy', dtype=np.int32, mode='r',
                            shape=(self.dstore_sizes[dstore_idx-1], 1))

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if self.move_dstore_to_mem:
            logger.info('Loading to memory...')
            start = time.time()

            if not self.no_load_keys:
                del keys
                self.keys_from_memmap = np.memmap(f'{keys_vals_prefix}_keys.npy', 
                    dtype=np.float16, mode='r', shape=(self.dstore_sizes[dstore_idx-1], self.dimension))
                keys = self.keys_from_memmap[:].astype(np.float16)
            else:
                keys = None

            del vals
            vals_from_memmap = np.memmap(f'{keys_vals_prefix}_vals.npy', dtype=np.int32, mode='r', shape=(self.dstore_sizes[dstore_idx-1], 1))
            vals = torch.from_numpy(vals_from_memmap[:]).long().to(self.device)
            del vals_from_memmap
            logger.info('Loading to memory took {} s'.format(time.time() - start))

        return cpu_index, gpu_index, keys, vals

    def setup_on_the_fly(self, idx):
        otf_prefix = get_otf_dstore_path(self.dstore_dir, self.model.config.model_type, self.dstore_sizes[idx-1], self.dimension, idx)
        Path(otf_prefix).parent.mkdir(parents=True, exist_ok=True)
        self.keys = np.memmap(f'{otf_prefix}_keys.npy', dtype=np.float16, mode='w+',
                              shape=(self.dstore_sizes[idx-1], self.dimension))
        self.vals = np.memmap(f'{otf_prefix}_vals.npy', dtype=np.int32, mode='w+',
                          shape=(self.dstore_sizes[idx-1], 1))

        #create a onthefly index to be changed and to keep original index
        otf_index_name = get_otf_index_path(self.dstore_dir, self.model.config.model_type, self.dstore_sizes[idx-1], self.dimension, idx)
        index_name = get_index_path(self.dstore_dir, self.model.config.model_type, self.dstore_sizes[idx-1], self.dimension, idx)
        shutil.copy(index_name, otf_index_name)

    def break_into(self, model):
        self.model = model
        model.broken_into = True

        self.reconstruct_index = []
        self.index = []
        self.keys = []
        self.vals = []
        for i in range(self.dstore_num):
            _, index_i, keys, vals = self.setup_faiss(i+1)
            self.index.append(index_i)
            if keys:
                self.keys.append(keys)
            self.vals.append(vals)
        self.index = tuple(self.index)
        self.keys, self.vals = tuple(self.keys), tuple(self.vals)

        if self.on_the_fly:
            self.idx_to_feedback = 1
            self.setup_on_the_fly(self.idx_to_feedback)
        
        self.is_encoder_decoder = model.config.is_encoder_decoder

        # Inject our pre_forward_hook to capture the labels at every forward pass
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook
        
        # Inject our activation_capturer to capture the activations at every forward pass
        layer_to_capture_fn, capture_input = KNNWrapper.model_layer_to_capture[model.config.model_type][self.knn_keytype]
        layer_to_capture = layer_to_capture_fn(model)
        self.activation_capturer = ActivationCapturer(layer_to_capture, capture_input=capture_input)
        self.register_hook(layer_to_capture, self.activation_capturer)

        # Inject our main function after the model's final layer
        final_layer = KNNWrapper.get_model_last_layer(model.config.model_type)(model)
        self.register_hook(final_layer, self.post_forward_hook)
        self.vocab_size = final_layer.out_features

    def get_knns(self, queries, idx):
        if not self.knn_gpu:
            queries = queries.cpu()
        dists_i, knns_i = self.index[idx].search(queries, self.k)
        dists_i, knns_i = dists_i.to(self.device), knns_i.to(self.device)
        return dists_i, knns_i

    def update_index(self, which_datastore, num_keys_to_add_at_a_time=500):
        """
        """
        index_name = get_otf_index_path(self.dstore_dir, self.model.config.model_type, self.dstore_sizes[which_datastore-1], self.dimension, which_datastore) 

        logger.info('Adding Keys')
        start = self.dstore_idx
        start_time = time.time()
        while start < self.dstore_sizes[which_datastore-1]:
            end = min(self.dstore_sizes[which_datastore-1], start + num_keys_to_add_at_a_time)
            to_add = self.keys[start:end].copy()
            self.index.add_with_ids(torch.tensor(to_add.astype(np.float32)), torch.arange(start, end))
            start += num_keys_to_add_at_a_time
            added_keys = start-self.dstore_idx
            if (start % num_keys_to_add_at_a_time) == 0:
                logger.info(f'Added {added_keys} tokens so far')
                logger.info(f'Writing Index {added_keys}')
                faiss.write_index(self.index, f'{index_name}')
        
        logger.info(f'Adding total {added_keys} keys')
        logger.info(f'Adding took {time.time() - start_time} s')
        logger.info(f'Writing Index to {index_name}')
        start = time.time()
        faiss.write_index(self.index, f'{index_name}')
        logger.info(f'Writing index took {time.time() - start} s')
        return 0
    
    def add_refs_datastore(self, which_datastore, keys, values):
        """
        """
        #number of tokens do add to datastore
        batch_time_size = keys.shape[0]

        #increase datastore size
        old_size = self.dstore_sizes[which_datastore-1]
        self.dstore_sizes[which_datastore-1]  += batch_time_size
        otf_prefix = get_otf_dstore_path(self.dstore_dir, self.model.config.model_type, old_size, self.dimension, which_datastore)

        self.keys = np.memmap(f'{otf_prefix}_keys.npy', dtype=np.float16, mode='r+',
                                  shape=(self.dstore_sizes[which_datastore-1], self.dimension))
        self.vals = np.memmap(f'{otf_prefix}_vals.npy', dtype=np.int32, mode='r+',
                              shape=(self.dstore_sizes[which_datastore-1], 1))
        
        try:
            self.keys[self.dstore_idx:(batch_time_size + self.dstore_idx)] = keys.cpu().numpy().astype(np.float16)
            self.vals[self.dstore_idx:(batch_time_size + self.dstore_idx)] = values.unsqueeze(-1).cpu().numpy().astype(np.int32)
            self.vals = torch.from_numpy(self.vals).to(self.device)
        except ValueError as ex:
            logger.error(f'Error saving datastore with mode {self.dstore_keys.mode}, did you try to save an already existing datastore?')
            logger.error(f'Delete the files {self.dstore_keys.filename} and {self.dstore_vals.filename} and try again')
            raise ex
    
        self.update_index(which_datastore=which_datastore)
        new_name = get_otf_dstore_path(self.dstore_dir, self.model.config.model_type, self.dstore_sizes[which_datastore-1], self.dimension, which_datastore)
        #Update dstore size of keys and values
        os.rename(f'{otf_prefix}_keys.npy', f'{new_name}_keys.npy')
        os.rename(f'{otf_prefix}_vals.npy', f'{new_name}_vals.npy')

        #delete old faiss index
        old_index_name = get_otf_index_path(self.dstore_dir, self.model.config.model_type, old_size, self.dimension, which_datastore) 
        if os.path.isfile(old_index_name):
            os.remove(old_index_name)
        else:
            # If it fails, inform the user.
            logger.info(f"Error: {old_index_name} file not found")
        
        self.dstore_idx += batch_time_size
        return 0

    def get_refs_forward(self):
        """
        """
        shift = 0 if self.is_encoder_decoder else 1
        captured_keys = self.activation_capturer.captured.to(self.device)
        if shift == 1:
            captured_keys = captured_keys[:, :-shift]
        captured_keys = captured_keys.flatten(0, 1) # (batch * time, dim)
        captured_values = self.labels[:, shift:].flatten(0, 1).to(self.device) # (batch * time)
    
        nonpad_mask = captured_values != -100
        keys = captured_keys[nonpad_mask]
        values = captured_values[nonpad_mask]
        return keys, values

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        self.labels = labels
        return self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)

    def post_forward_hook(self, module, input, output):
        batch, time_dim, vocab_size = output.shape
        shift = 0 if self.is_encoder_decoder else 1
        lm_logits = output
        lm_logits = torch.nn.functional.log_softmax(lm_logits, dim=-1).to(self.device) # (batch, time, vocab)
        queries = self.activation_capturer.captured # (batch, time, dim)

        if self.labels is None:
            nonpad_mask = torch.cat([
                torch.zeros([batch, time_dim - 1], dtype=torch.bool),
                torch.ones([batch, 1], dtype=torch.bool),
            ], axis=-1).to(self.device)
        else:
            nonpad_mask = torch.cat([
                self.labels[:, shift:] != -100, 
                torch.zeros([self.labels.shape[0], shift], dtype=torch.bool).to(self.device)
            ], axis=-1)

        lm_logits = lm_logits[nonpad_mask]
        queries = queries[nonpad_mask] # (nonpad, dim)
        
        dists, knns = [], []
        for i in range(self.dstore_num):
            dists_i, knns_i = self.get_knns(queries, i) # (nonpad batch * time, k)
            dists.append(dists_i)
            knns.append(knns_i)
        dists = tuple(dists)
        knns = tuple(knns)

        if self.recompute_dists:
            dists = []
            for i in range(self.dstore_num):
                knns_vecs = torch.from_numpy(self.keys[i][knns[i]]).to(self.device)
                dists.append(self.dist_func(queries, knns_vecs))

        neg_dists = []
        for i in range(self.dstore_num):
            neg_dists.append(-dists[i])
        neg_dists = tuple(neg_dists)

        knn_log_probs = []
        for i in range(self.dstore_num):
            knn_log_probs_i, _ = self.knns_to_log_prob(knns[i], neg_dists[i], self.vals[i])
            knn_log_probs.append(knn_log_probs_i)

        interpolated_scores = KNNWrapper.interpolate(knn_log_probs, lm_logits, self.lmbdas, self.dstore_num) # (nonpad, vocab)
        output[nonpad_mask] = interpolated_scores

        #Perform on-the-fly human feedback loop
        if self.on_the_fly:
            if self.labels is not None:
                keys, values = self.get_refs_forward()
                self.add_refs_datastore(self.idx_to_feedback, keys, values)

        return output 

    def knns_to_log_prob(self, knns, neg_dists, vals): 
        probs = torch.nn.functional.softmax(neg_dists / self.knn_temperature, dim=-1)
        vals_at_knns = vals[knns].squeeze(-1).to(torch.int64) #(nonpad batch * time, k)
        knn_log_probs = torch.full(size=(vals_at_knns.shape[:-1] + (self.vocab_size,)), fill_value=0.0).to(self.device) \
            .scatter_add(dim=-1, index=vals_at_knns, src=probs).log() # (nonpad_batch * time, vocab)
        knn_log_probs = torch.nan_to_num(knn_log_probs, nan=None, neginf=-10000.0)
        return knn_log_probs, vals_at_knns
        
    def register_hook(self, layer, func, pre=False):
        handle = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(handle)

    def break_out(self):
        for h in self.hook_handles:
            h.remove()
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None
    
    def get_metrics(self):
        return {}
    
    @staticmethod
    def l2(query, keys):
        # query: (batch*time, dim)
        # keys:  (batch*time, k, dim)
        # returns: (batch*time, k)
        return torch.sum((query.unsqueeze(-2) - keys)**2, dim=-1)

    @staticmethod
    def dotprod(query, keys):
        # query: (batch, beams, dim)
        # keys:  (batch, 1, time, dim)
        # returns: (batch, beams, time)
        return torch.sum((query.unsqueeze(-2) * keys), dim=-1)

    @staticmethod
    def interpolate(knn_log_probs, lm_log_probs, lmbdas, dstore_num):
        knn_weighted_log_probs = 0
        for i in range(dstore_num):
            knn_weighted_log_probs += torch.exp(knn_log_probs[i] + torch.log(lmbdas[i][0]))

        interpolated = torch.log(torch.exp(lm_log_probs + torch.log(1 - torch.sum(lmbdas, dim=0))) + knn_weighted_log_probs)
        
        return interpolated

    @staticmethod
    def get_model_last_layer(model_type):
        # works for gpt2, marian, t5. If a model does not have an ".lm_head" layer, 
        # add an "if model_type is ..." statement here, and return the output embedding layer
        return lambda model: model.lm_head

    @staticmethod
    def get_model_embedding_layer(model_type):
        if model_type.startswith('gpt2'):
            return lambda model: model.transformer.wte

    # For every model name and key type, returns a lambda that returns the relevant layer in the model, 
    # and whether the input of that layer should be captured (True) or the output (False)
    model_layer_to_capture = {
        'bart': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.layers[-1].fc1, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.layers[-1], False),
        },
        'gpt2': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.h[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.h[-1], False),
        },
        'marian': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.layers[-1].fc1, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.layers[-1], False),
        },
        't5': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.block[-1].layer[2].DenseReluDense, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.block[-1].layer[2], False),
        }
}
    
class KNNSaver(object):
    def __init__(self, dstore_size, dstore_dir, dimension, knn_keytype=None, dstore_num=1):
        self.dstore_size = dstore_size
        self.dstore_dir = dstore_dir
        self.dimension = dimension
        self.knn_keytype = KEY_TYPE.last_ffn_input if knn_keytype is None else knn_keytype

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.activation_capturer = None
        self.is_encoder_decoder = None
        self.dstore_idx = 0
        self.dstore_keys = None
        self.dstore_vals = None
        self.labels = None
        self.hook_handles = []
        self.dstore_num = dstore_num

        logger.info(f'keytype being saved: {self.knn_keytype}')
        logger.info('Saving fp16')

    def break_into(self, model):
        self.model = model
        model.broken_into = True
        self.is_encoder_decoder = model.config.is_encoder_decoder
        
        # Inject our activation_capturer to capture the activations at every forward pass
        layer_to_capture_fn, capture_input = KNNWrapper.model_layer_to_capture[model.config.model_type][self.knn_keytype]
        layer_to_capture = layer_to_capture_fn(model)
        self.activation_capturer = ActivationCapturer(layer_to_capture, capture_input=capture_input)
        self.register_hook(layer_to_capture, self.activation_capturer)

        # Inject our pre_forward_hook to capture the labels at every forward pass
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook
        
        # Inject our main function after the model's final layer
        final_layer = KNNWrapper.get_model_last_layer(model.config.model_type)(model)
        self.register_hook(final_layer, self.post_forward_hook)

        keys_vals_prefix = get_dstore_path(self.dstore_dir, model.config.model_type, self.dstore_size, self.dimension, self.dstore_num)
        keys_filename = f'{keys_vals_prefix}_keys.npy'
        vals_filename = f'{keys_vals_prefix}_vals.npy'
        if os.path.exists(keys_filename) and os.path.exists(vals_filename):
            mode = 'r'
        else:
            mode = 'w+'
            Path(keys_filename).parent.mkdir(parents=True, exist_ok=True)
        
        self.dstore_keys = np.memmap(keys_filename, dtype=np.float16, mode=mode, shape=(self.dstore_size, self.dimension))
        self.dstore_vals = np.memmap(vals_filename, dtype=np.int32, mode=mode, shape=(self.dstore_size, 1))

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if labels is None:
            raise ValueError('labels must be provided when saving a datastore. Are you using --predict_with_generate by mistake? If so, disable it')
        self.labels = labels
        return self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)

    def post_forward_hook(self, module, input, output):
        shift = 0 if self.is_encoder_decoder else 1
        captured_keys = self.activation_capturer.captured.to(self.device)
        if shift == 1:
            captured_keys = captured_keys[:, :-shift]
        captured_keys = captured_keys.flatten(0, 1) # (batch * time, dim)
        captured_values = self.labels[:, shift:].flatten(0, 1).to(self.device) # (batch * time)

        nonpad_mask = captured_values != -100
        keys = captured_keys[nonpad_mask]
        values = captured_values[nonpad_mask]

        batch_time_size = keys.shape[0]

        # if shape[0] == args.tokens_per_sample:
        if self.dstore_idx + batch_time_size > self.dstore_size:
            batch_time_size = max(self.dstore_size - self.dstore_idx, 0)
            keys = keys[:batch_time_size]
            values = values[:batch_time_size]
        try:
            self.dstore_keys[self.dstore_idx:(batch_time_size + self.dstore_idx)] = keys.cpu().numpy().astype(np.float16)
            self.dstore_vals[self.dstore_idx:(batch_time_size + self.dstore_idx)] = values.unsqueeze(-1).cpu().numpy().astype(np.int32)
        except ValueError as ex:
            logger.error(f'Error saving datastore with mode {self.dstore_keys.mode}, did you try to save an already existing datastore?')
            logger.error(f'Delete the files {self.dstore_keys.filename} and {self.dstore_vals.filename} and try again')
            raise ex

        self.dstore_idx += batch_time_size
        
        return output

    def register_hook(self, layer, func, pre=False):
        handle = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(handle)
    
    def break_out(self):
        for h in self.hook_handles:
            h.remove()
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None

    def build_index(self, num_keys_to_add_at_a_time=1000000, 
            ncentroids=4096, seed=1, code_size=64, probe=32):
        logger.info('Building index')
        index_name = get_index_path(self.dstore_dir, self.model.config.model_type, self.dstore_size, self.dimension, self.dstore_num) 
        
        # Initialize faiss index
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFPQ(quantizer, self.dimension,
            ncentroids, code_size, 8)
        index.nprobe = probe

        logger.info('Training Index')
        np.random.seed(seed)
        random_sample = np.random.choice(np.arange(self.dstore_vals.shape[0]), size=[min(1000000, self.dstore_vals.shape[0])], replace=False)
        start = time.time()
        # Faiss does not handle adding keys in fp16 as of writing this.
        index.train(self.dstore_keys[random_sample].astype(np.float32))
        logger.info(f'Training took {time.time() - start} s')

        logger.info('Adding Keys')
        # index = faiss.read_index(f'{index_name}.trained')
        start = 0
        start_time = time.time()
        while start < self.dstore_size:
            end = min(self.dstore_size, start + num_keys_to_add_at_a_time)
            to_add = self.dstore_keys[start:end].copy()
            index.add_with_ids(torch.tensor(to_add.astype(np.float32)), torch.arange(start, end))
            start += num_keys_to_add_at_a_time

            if (start % 1000000) == 0:
                logger.info(f'Added {start} tokens so far')
                logger.info(f'Writing Index {start}')
                faiss.write_index(index, f'{index_name}')

        logger.info(f'Adding total {start} keys')
        logger.info(f'Adding took {time.time() - start_time} s')
        logger.info(f'Writing Index to {index_name}')
        start = time.time()
        faiss.write_index(index, f'{index_name}')
        logger.info(f'Writing index took {time.time() - start} s')
        
    def get_metrics(self):
        return {}

class KNNCorrections(object):
    def __init__(self, dstore_dir, dstore_size, dimension, knn_keytype=None, probe=32, dstore_num=1):
        self.dstore_dir = dstore_dir
        self.dstore_size = dstore_size
        self.dimension = dimension
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.knn_keytype = KEY_TYPE.last_ffn_input if knn_keytype is None else knn_keytype

        self.model = None
        self.activation_capturer = None
        self.is_encoder_decoder = None
        self.dstore_idx = self.dstore_size
        self.dstore_keys = None
        self.dstore_vals = None
        self.labels = None
        self.hook_handles = []
        self.dstore_num = dstore_num

        self.probe = probe

        #If using gpu, a cpu index is needed because it's not possible to call add_with_ids for
        #index in gpu as of this moment: https://github.com/facebookresearch/faiss/pull/2263
        #This means tat on-the-fly can't be used with indexes in gpu as of yet
        self.knn_gpu = False

        logger.info(f'keytype being saved: {self.knn_keytype}')
        logger.info('Saving fp16')

    def setup_faiss(self):
        if not self.dstore_dir:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()
        index_name = get_index_path(self.dstore_dir, self.model.config.model_type, self.dstore_size, self.dimension, self.dstore_num) 
        cpu_index = faiss.read_index(index_name, faiss.IO_FLAG_ONDISK_SAME_DIR)
        logger.info(f'Reading datastore took {time.time() - start} s')
        cpu_index.nprobe = self.probe

        if self.knn_gpu:
            start = time.time()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index, co)
            logger.info(f'Moving index to GPU took {time.time() - start} s')
        else:
            gpu_index = cpu_index

        return cpu_index, gpu_index

    def break_into(self, model):
        self.model = model
        model.broken_into = True
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.reconstruct_index, self.index = self.setup_faiss()
        self.is_encoder_decoder = model.config.is_encoder_decoder
        
        # Inject our activation_capturer to capture the activations at every forward pass
        layer_to_capture_fn, capture_input = KNNWrapper.model_layer_to_capture[model.config.model_type][self.knn_keytype]
        layer_to_capture = layer_to_capture_fn(model)
        self.activation_capturer = ActivationCapturer(layer_to_capture, capture_input=capture_input)
        self.register_hook(layer_to_capture, self.activation_capturer)

        # Inject our pre_forward_hook to capture the labels at every forward pass
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook
        
        # Inject our main function after the model's final layer
        final_layer = KNNWrapper.get_model_last_layer(model.config.model_type)(model)
        self.register_hook(final_layer, self.post_forward_hook)

        #keys_vals_prefix = get_dstore_path(self.dstore_dir, model.config.model_type, self.dstore_size, self.dimension)
        #keys_filename = f'{keys_vals_prefix}_keys.npy'
        #vals_filename = f'{keys_vals_prefix}_vals.npy'
        
        #self.dstore_keys = np.memmap(keys_filename, dtype=np.float16, mode='r+', shape=(self.dstore_size, self.dimension))
        #self.dstore_vals = np.memmap(vals_filename, dtype=np.int32, mode='r+', shape=(self.dstore_size, 1))

    def update_index(self, batch_time_size, num_keys_to_add_at_a_time=500):
        """
        """
        index_name = get_index_path(self.dstore_dir, self.model.config.model_type, self.dstore_size, self.dimension, self.dstore_num) 

        logger.info('Adding Keys')
        start = self.dstore_idx
        start_time = time.time()
        while start < self.dstore_size:
            end = min(self.dstore_size, start + num_keys_to_add_at_a_time)
            to_add = self.dstore_keys[start:end].copy()
            self.index.add_with_ids(torch.tensor(to_add.astype(np.float32)), torch.arange(start, end))
            start += num_keys_to_add_at_a_time
            added_keys = start-self.dstore_idx
            if (start % num_keys_to_add_at_a_time) == 0:
                logger.info(f'Added {added_keys} tokens so far')
                logger.info(f'Writing Index {added_keys}')
                faiss.write_index(self.index, f'{index_name}')
        
        logger.info(f'Adding total {added_keys} keys')
        logger.info(f'Adding took {time.time() - start_time} s')
        logger.info(f'Writing Index to {index_name}')
        start = time.time()
        faiss.write_index(self.index, f'{index_name}')
        logger.info(f'Writing index took {time.time() - start} s')
        return 0
    
    def add_refs_datastore(self, datastore_dir, keys, values):
        """
        """
        #number of tokens do add to datastore
        batch_time_size = keys.shape[0]

        #increase datastore size
        old_size = self.dstore_size
        self.dstore_size  += batch_time_size
        keys_vals_prefix = get_dstore_path(self.dstore_dir, self.model.config.model_type, old_size, self.dimension, self.dstore_num)

        self.dstore_keys = np.memmap(f'{keys_vals_prefix}_keys.npy', dtype=np.float16, mode='r+',
                                  shape=(self.dstore_size, self.dimension))
        self.dstore_vals = np.memmap(f'{keys_vals_prefix}_vals.npy', dtype=np.int32, mode='r+',
                              shape=(self.dstore_size, 1))
        
        try:
            self.dstore_keys[self.dstore_idx:(batch_time_size + self.dstore_idx)] = keys.cpu().numpy().astype(np.float16)
            self.dstore_vals[self.dstore_idx:(batch_time_size + self.dstore_idx)] = values.unsqueeze(-1).cpu().numpy().astype(np.int32)
            self.dstore_vals = torch.from_numpy(self.dstore_vals).to(self.device)
        except ValueError as ex:
            logger.error(f'Error saving datastore with mode {self.dstore_keys.mode}, did you try to save an already existing datastore?')
            logger.error(f'Delete the files {self.dstore_keys.filename} and {self.dstore_vals.filename} and try again')
            raise ex
    
        self.update_index(batch_time_size=batch_time_size)
        new_name = get_dstore_path(self.dstore_dir, self.model.config.model_type, self.dstore_size, self.dimension,self.dstore_num)
        #Update dstore size of keys and values
        os.rename(f'{keys_vals_prefix}_keys.npy', f'{new_name}_keys.npy')
        os.rename(f'{keys_vals_prefix}_vals.npy', f'{new_name}_vals.npy')

        #delete old faiss index
        old_index_name = get_index_path(self.dstore_dir, self.model.config.model_type, old_size, self.dimension, self.dstore_num) 
        if os.path.isfile(old_index_name):
            os.remove(old_index_name)
        else:
            # If it fails, inform the user.
            logger.info(f"Error: {old_index_name} file not found")
        
        self.dstore_idx += batch_time_size
        return 0

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if labels is None:
            raise ValueError('labels must be provided when saving a datastore. Are you using --predict_with_generate by mistake? If so, disable it')
        self.labels = labels
        return self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)

    def post_forward_hook(self, module, input, output):
        shift = 0 if self.is_encoder_decoder else 1
        captured_keys = self.activation_capturer.captured.to(self.device)
        if shift == 1:
            captured_keys = captured_keys[:, :-shift]
        captured_keys = captured_keys.flatten(0, 1) # (batch * time, dim)
        captured_values = self.labels[:, shift:].flatten(0, 1).to(self.device) # (batch * time)

        nonpad_mask = captured_values != -100
        keys = captured_keys[nonpad_mask]
        values = captured_values[nonpad_mask]

        batch_time_size = keys.shape[0]
        
        self.add_refs_datastore(None, keys, values)
        self.dstore_idx += batch_time_size
        
        return output

    def register_hook(self, layer, func, pre=False):
        handle = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(handle)
    
    def break_out(self):
        for h in self.hook_handles:
            h.remove()
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None



class ActivationCapturer(nn.Module):
    def __init__(self, layer, capture_input=False):
        super().__init__()
        self.layer = layer
        self.capture_input = capture_input

        self.captured = None
  
    def forward(self, module, input, output):
        if self.capture_input:
            self.captured = input[0].detach()
        else:
            self.captured = output.detach()

def get_dstore_path(dstore_dir, model_type, dstore_size, dimension, dstore_num):
    return f'{dstore_dir}/dstore{dstore_num}_{model_type}_{dstore_size}_{dimension}'

def get_otf_dstore_path(dstore_dir, model_type, dstore_size, dimension, dstore_num):
    return f'{dstore_dir}/on-the-fly/dstore{dstore_num}_{model_type}_{dstore_size}_{dimension}'

def get_index_path(dstore_dir, model_type, dstore_size, dimension, dstore_num):
    return f'{dstore_dir}/index{dstore_num}_{model_type}_{dstore_size}_{dimension}.indexed'

def get_otf_index_path(dstore_dir, model_type, dstore_size, dimension, dstore_num):
    return f'{dstore_dir}/on-the-fly/index{dstore_num}_{model_type}_{dstore_size}_{dimension}.indexed'