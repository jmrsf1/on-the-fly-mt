from os import listdir
from os.path import isfile, join
import json

import argparse


def retrieve_and_preprocess_datasets(basepath="/Users/joaofonseca/Documents/!TESE/datasets/lv-en/"):
    """
    Retrieves datasets from basepath folder. basepath must contain 2 text files with format: "name.src.txt" and "name.tgt.txt"
    where src and tgt are codes of source and target language respectively (e.g. "en" for English and "es" for Spanish).
    """
    dataset_paths = [f for f in listdir(basepath) if isfile(join(basepath, f))]
    source_language  = dataset_paths[0].split(".")[1].split("-")[0]
    target_language  = dataset_paths[0].split(".")[1].split("-")[1]

    if dataset_paths[0].split(".")[2] == source_language:
        source_file = basepath+dataset_paths[0]
        target_file = basepath+dataset_paths[1]
    else:
        source_file = basepath+dataset_paths[1]
        target_file = basepath+dataset_paths[0]

    text_to_json(source_file, target_file, source_language, target_language)
    return 0

def text_to_json(filename_src:str, filename_trg:str, source_lang:str, target_lang:str):
    """
    Converts 2 text files with source language and target language respectively to a json 
    file with the following format:
    {
        "translation" : {
            "<src>" : "<source_text>",
            "<tgt>" : "<target_text>"
        }
    }
    """
    translations = []
    with open(filename_src) as f_src, open(filename_trg, encoding='utf-8') as f_trg:
        content_src = f_src.readlines()
        content_trg = f_trg.readlines()
    for i in range(len(content_src)):
        translation = {}
        translation[source_lang] = content_src[i]
        translation[target_lang] = content_trg[i]
        translations.append({"translation" : translation})
    
    #base path translation
    basepath = filename_src.split(".")[0] + "." + filename_src.split(".")[1]

    # the json file where the output must be stored
    out_file = open(basepath + ".json", "w")
    json.dump(translations, out_file, indent=2, ensure_ascii=False)
    out_file.close()

    return 1


def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-b", "--basepath", help="Directory where the datasets are stored")
    args = argParser.parse_args()

    _ = retrieve_and_preprocess_datasets(args.basepath)
    

if __name__ == "__main__":
    main()
