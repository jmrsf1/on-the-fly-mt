from os import listdir
from os.path import isfile, join
import json

def retrieve_europarl_datasets(basepath="/Users/joaofonseca/Documents/!TESE/datasets/lv-en/"):
    dataset_paths = [f for f in listdir(basepath) if isfile(join(basepath, f))]
    extensions_src = dataset_paths[0].split(".")
    extensions_trg = dataset_paths[1].split(".")

    text_to_json(basepath+dataset_paths[0], basepath+dataset_paths[1], extensions_src[2], extensions_trg[2])
    return

def text_to_json(filename_src:str, filename_trg:str, source_lang:str, target_lang:str):
    translations = []
    with open(filename_src) as f_src, open(filename_trg) as f_trg:
        content_src = f_src.readlines()
        content_trg = f_trg.readlines()
    for i in range(len(content_src)):
        translation = {}
        translation[source_lang] = content_src[i]
        translation[target_lang] = content_trg[i]
        translations.append({"translation" : translation})
        if i >=100:
            break
    
    #base path of europarl translation
    basepath = filename_src.split(".")[0] + "." + filename_src.split(".")[1]

    # the json file where the output must be stored
    out_file = open(basepath + ".json", "w")
    json.dump(translations, out_file)
    out_file.close()

    return 1


def main():
    #path of 
    basepath = "/Users/joaofonseca/Documents/!TESE/datasets/lv-en/"
    retrieve_europarl_datasets(basepath)
    

if __name__ == "__main__":
    main()
