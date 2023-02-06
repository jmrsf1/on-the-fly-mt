
import os
from knnlm import KNNWrapper, KNNSaver, KEY_TYPE, DIST
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



def load_model():
    """
    Loads model and tokenizer from huggingface
    """
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-mbart")
    model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/tiny-mbart")
    return model, tokenizer






def main():
    model, tokenizer = load_model()



    #knn_wrapper = KNNWrapper(...)
    #knn_wrapper.break_into(model)
    pass


if __name__ == "__main__":
    main()
