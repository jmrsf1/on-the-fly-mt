# On-the-fly Machine Translation

<img
  src="https://github.com/jmrsf1/on-the-fly-mt/blob/main/images/on-the-fly.png"
  alt="Alt text"
  title="On-the-fly Machine Translation"
  style="display: block; width: 425px;">

Added functionalities of using custom dataset, using and creating multiple datastores, perform on-the-fly machine translation (i.e., add a corrections dataset to one of the datastores) or perform a 'on-the-fly inference' which, for every performed translation, adds the correspondent references to the datastore on-the-fly to be applied when inference has a sequential nature.

#### Multiple Datastores: 
-> Create datastores offline (use <dstore_num> equal to 1 when creating the first datastore, <dstore_num> equal to 2 when creating the second one and so on). Different datastores will have this <dstore_num> variable in their name (serving as an index).
-> Provide <lmbdas> variable in the following format: <[[λ1],[λ2],[λ3],...]> (a weight to each datastore).
-> When in inference, the variable <dstore_num> is the number of datastores to be used (if it is equal to 3, datastores 1,2 and 3 will be considered).
-> All datastores scores are linearly interpolated with the model's prediction using the given weights.
-> Provide <dstore_sizes> (instead of <dstore_size>) in the format: <[size_ds1,size_ds2,size_ds3,...]>

## 1) Preprocess custom dataset (if not using huffing face datasets)
-> Use script to transform custom text parallel corpus (2 .txt files with source 
and target corpus respectively) to the following json format:
 [ { "translation" : "<src>": "...", "<trg>": "..." }, ... ]

``` python3 dataset_preprocessing.py --basepath </path/to/datasets> ```


## 2) Save a datastore to <dstore_dir>.
-> <dstore_num> is the index of the current datastore being added. If the first datastore is being created, <dstore_num> should be 1, and so on.
-> <dstore_size> is the size of the current datastore being added.
-> <eval_subset> is the subset to be added to datastore.

``` 
  CUDA_VISIBLE_DEVICES=0 python3 -u run_translation.py  \
  --model_name_or_path t5-small  \
  --dataset_name wmt16 --dataset_config_name ro-en \
  --per_device_train_batch_size 4 --per_device_eval_batch_size=4 \
  --output_dir checkpoints-translation/t5-small \
  --source_lang en --target_lang ro \
  --dstore_size 85108 --dstore_num 1 \
  --dstore_dir checkpoints-translation/t5-small \
   --save_knnmt_dstore --do_eval --eval_subset validation \
   --source_prefix "translate English to Romanian: "
  ```

## 2.5) Save a datastore but using custom json validation file 

``` 
  python3 -u run_translation.py  \
  --model_name_or_path t5-small  \
  --validation_file ../datasets/lv-en/europarl-v7.lv-en.json --dataset_config_name ro-en \
  --per_device_train_batch_size 4 --per_device_eval_batch_size=4 \
  --output_dir checkpoints-translation/t5-small \
  --source_lang en --target_lang ro \
  --dstore_size 85108 --dstore_num 1 \
  --dstore_dir checkpoints-translation/t5-small \
   --save_knnlm_dstore --do_eval \
   --source_prefix "translate English to Romanian: "
  ```


## 3) Build faiss index for datastore <dstore_num>

```
  python3 -u run_translation.py  \
  --model_name_or_path t5-small \
  --dataset_name wmt16 --dataset_config_name ro-en \
  --per_device_train_batch_size 4 --per_device_eval_batch_size=4 \
  --output_dir checkpoints-translation/t5-small \
  --source_lang en --target_lang ro \
  --dstore_size 85108 --dstore_num 1 \
  --dstore_dir checkpoints-translation/t5-small \
  --build_index 
  ```


## 4) Inference using all datastores up to <dstore_num>
-> <lmbdas> variable is of the following format: <[[λ1],[λ2],[λ3],...]> (no spaces) according to <dstore_num> (if <dstore_num> is 2, you must provide 'lmbdas' as <[[λ1],[λ2]]> )
-> When inferring, use <dstore_sizes> (instead of <dstore_size>) in the format: <[size_ds1,size_ds2,size_ds3,...]>.
```
  CUDA_VISIBLE_DEVICES=1 python -u run_translation.py  \
  --model_name_or_path t5-small \
  --dataset_name wmt16 --dataset_config_name ro-en \
  --per_device_eval_batch_size=4 \
  --output_dir checkpoints-translation/t5-small \
  --source_lang en --target_lang ro \
  --do_predict \
  --predict_with_generate \
  --source_prefix "translate English to Romanian: " \
  --dstore_sizes [85108,85108] --dstore_num 2 \
  --dstore_dir checkpoints-translation/t5-small \
  --knn_temp 50 --k 16 --lmbdas [[0.1],[0.8]] \
  --knn 
  ```

## 5) Inference On-the-fly
-> Every reference of every test set batch translated is inserted inside the datastore,
to mimic a human feedback loop [assuming that there are test set references] as soon as it's
machine translated.
-> The index and datastores are put on an <on-the-fly> named folder inside <dstore_dir>. To use this datastore
on future runs just copy datastore and index to <dstore_dir> after the run and delete on-the-fly.

```
  CUDA_VISIBLE_DEVICES=1 python -u run_translation.py  \
  --model_name_or_path t5-small \
  --dataset_name wmt16 --dataset_config_name ro-en \
  --per_device_eval_batch_size=4 \
  --output_dir checkpoints-translation/t5-small \
  --source_lang en --target_lang ro \
  --do_predict \
  --predict_with_generate \
  --source_prefix "translate English to Romanian: " \
  --dstore_sizes [85108] --dstore_num 1 \
  --dstore_dir checkpoints-translation/t5-small \
  --knn_temp 50 --k 16 --lmbdas [[0.25]] \
  --knn --on_the_fly
  ```

## 6) Add corrections to datastore
-> Provide the <corrections.'src'-'trg'.json> file (see dataset_preprocessing.py for more details about correct format) which should be inside of folder <corrections/>.
-> These corrections are added to the <dstore_num> datastore in dstore_dir, overwritting it
with new version with corrections included.
-> <dstore_size> is the size of the current datastore being added.

```
  CUDA_VISIBLE_DEVICES=1 python -u run_translation.py  \
  --model_name_or_path t5-small \
  --corrections /home/joaofonseca/on-the-fly-mt/corrections/corrections.en-ro.json --dataset_config_name ro-en \
  --per_device_eval_batch_size=4 \
  --output_dir checkpoints-translation/t5-small \
  --source_lang en --target_lang ro \
  --source_prefix "translate English to Romanian: " \
  --dstore_size 85108 --dstore_num 1\
  --dstore_dir checkpoints-translation/t5-small
  ```
  
  ## Citation

This repository implements:
-> [Nearest Neighbor Machine Translation: Nearest Neighbor Language Models](https://arxiv.org/abs/2010.00710)
```
@misc{khandelwalnnmt,
  doi = {10.48550/ARXIV.2010.00710},
  url = {https://arxiv.org/abs/2010.00710},
  author = {Khandelwal, Urvashi and Fan, Angela and Jurafsky, Dan and Zettlemoyer, Luke and Lewis, Mike},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Nearest Neighbor Machine Translation},
  publisher = {arXiv},
  year = {2020},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
  and is built-on the implementation in https://github.com/neulab/knn-transformers.


