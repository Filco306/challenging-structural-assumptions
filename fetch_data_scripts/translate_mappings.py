import argparse
import os
import sys
import pickle
from transformers import MarianMTModel, MarianTokenizer
from typing import List
import transformers
import time
import torch
from tqdm import tqdm
from dotenv import load_dotenv
import gc
from itertools import product
import pandas as pd
from pandas import json_normalize

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str, default="data/Mapping/total_entity2.p")
parser.add_argument("--batch_size", type=int, default=60)
parser.add_argument(
    "--export_filename", type=str, default="data/Mapping/total_entity_translated.p"
)
parser.add_argument("--lan_list", action="append")

# parser.add_argument('--append-action', action='append')
LANG = ["EN", "FR", "ES", "ZH", "JA", "AR", "DE", "SV", "RU", "PT", "NL", "IT", "EL"]
lang = set([item.lower() for item in LANG])
roman = ["fr", "es", "pt", "it"]
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_nan_cols(df, col="label"):
    english_cols = ["en", "en-ca", "en-gb"]
    return (
        df[[en + "." + t for en, t in product(english_cols, [col])]]
        .isnull()
        .all(axis=1)
        .sum()
    )


def get_model(src: str):
    model_ = "Helsinki-NLP/opus-mt-{src}-{tgt}".format(src=src, tgt="en")
    tokenizer = MarianTokenizer.from_pretrained(model_)
    model = MarianMTModel.from_pretrained(model_).to(device)
    return (tokenizer, model)


def get_language_index(lan_code: str):
    if lan_code in roman:
        return lan_index["roa"]
    elif lan_code in lan_index.keys():
        return lan_index[lan_code]
    else:
        return -1


def translate(
    texts: List[str],
    tokenizer,
    model,
    device="cpu",
):
    translated = model.generate(
        **tokenizer.prepare_seq2seq_batch(texts, return_tensors="pt").to(device)
    ).to(device)
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return tgt_text


def translate_entity(entity: dict):
    if "en" not in entity.keys():
        entity["en"] = {}
    models = {}
    if "label" not in entity["en"]:

        for key in entity.keys():
            index = get_language_index(key)
            if index != -1 and "label" in entity[key]:
                src_text = entity[key]["label"]
                tensor = (
                    models[index][1]
                    .generate(
                        **models[index][0](
                            [src_text], return_tensors="pt", padding=True
                        ).to(device)
                    )
                    .to(device)
                )
                entity["en"]["label"] = models[index][0].decode(
                    tensor[0], skip_special_tokens=True
                )
                break
        if "label" not in entity["en"]:
            entity["en"]["label"] = "Unknown"
    if "description" not in entity["en"]:

        for key in entity.keys():
            index = get_language_index(key)
            if index != -1 and "description" in entity[key]:
                src_text = entity[key]["description"]
                tensor = (
                    models[index][1]
                    .generate(
                        **models[index][0](
                            [src_text], return_tensors="pt", padding=True
                        ).to(device)
                    )
                    .to(device)
                )
                tgt_text = models[index][0].decode(tensor[0], skip_special_tokens=True)
                entity["en"]["description"] = tgt_text
                print(tgt_text)

                break
        if "description" not in entity["en"]:
            entity["en"]["description"] = "Unknown"

    return entity


def find_language(data, Id, label=True):
    entity = data[Id]
    for key in entity:
        if label:
            if ("label" in entity[key]) and get_language_index(key) != -1:
                return get_language_index(key), entity[key]["label"]

        else:
            if ("description" in entity[key]) and get_language_index(key) != -1:
                return get_language_index(key), entity[key]["description"]

    return -1, "Unknown"


def find_swedish(data, Id, label=True):
    entity = data[Id]
    for key in entity:
        if label:
            if ("label" in entity[key]) and get_language_index(key) != -1:
                if key == "sv":
                    return -2, entity[key]["label"]
                return get_language_index(key), entity[key]["label"]
        else:
            if ("description" in entity[key]) and get_language_index(key) != -1:
                if key == "sv":
                    return -2, entity[key]["description"]
                return get_language_index(key), entity[key]["description"]

    return -1, "Unknown"


def separate_data(data: dict):
    to_translate_label = []
    to_translate_des = []
    for key in data.keys():
        if "en" not in data[key]:
            to_translate_label.append(key)
            to_translate_des.append(key)
        else:
            if "label" not in data[key]["en"]:
                to_translate_label.append(key)
            if "description" not in data[key]["en"]:
                to_translate_des.append(key)
    label_translate = [[] for i in range(len(lan_list) + 1)]
    label_text = [[] for i in range(len(lan_list) + 1)]
    for Id in to_translate_label:
        lanID, text = find_language(data, Id, label=True)
        label_translate[lanID].append(Id)
        label_text[lanID].append(text)

    des_translate = [[] for i in range(len(lan_list) + 1)]
    des_text = [[] for i in range(len(lan_list) + 1)]
    for Id in to_translate_des:
        lanID, text = find_language(data, Id, label=False)
        des_translate[lanID].append(Id)
        des_text[lanID].append(text)

    return (label_translate, label_text), (des_translate, des_text)


def merge_dicts(init_dict, res_dict):
    for key in res_dict:
        # Iter over all the new results
        for label_desc in res_dict[key]["en"]:
            if key not in init_dict:
                init_dict[key] = {}
            if "en" not in init_dict[key]:
                init_dict[key]["en"] = {}
            init_dict[key]["en"][label_desc] = res_dict[key]["en"][label_desc]
    return init_dict


def get_swedish_data(data: dict):
    """For those classified as swedish """
    to_translate_label = []
    to_translate_des = []
    for key in data.keys():
        if "en" not in data[key]:
            to_translate_label.append(key)
            to_translate_des.append(key)
        else:
            if "label" not in data[key]["en"]:
                to_translate_label.append(key)
            if "description" not in data[key]["en"]:
                to_translate_des.append(key)

    label_translate = []
    label_text = []
    for Id in to_translate_label:
        lanID, text = find_swedish(data, Id, label=True)
        if lanID == -2:
            label_text.append(text)
            label_translate.append(Id)

    des_translate = []
    des_text = []
    for Id in to_translate_des:
        lanID, text = find_swedish(data, Id, label=False)
        if lanID == -2:
            des_text.append(text)
            des_translate.append(Id)
    return (label_translate, label_text), (des_translate, des_text)


def translate_mappings(args):
    file_path = args["file_path"]
    data = pickle.load(open(file_path, "rb"))
    if os.path.exists(os.path.join(args["export_filename"])) is True:
        data = merge_dicts(
            init_dict=data, res_dict=pickle.load(open(args["export_filename"], "rb"))
        )
    batch_size = args["batch_size"]

    label_trans, des_trans = separate_data(data)
    del data
    data = {}
    gc.collect()
    for i in range(len(lan_list)):

        ids = label_trans[0][i]
        texts = label_trans[1][i]
        if len(ids) > 0:
            print("Now processing :{}, with length{}".format(lan_list[i], len(ids)))
            (tokenizer, model) = get_model(lan_list[i])
            for k in tqdm(range(0, len(ids), batch_size)):
                id_ = ids[k : min(k + batch_size, len(ids))]
                texts_ = texts[k : min(k + batch_size, len(texts))]

                tgt_text = translate(texts_, tokenizer, model, device=device)
                for index, ID in enumerate(id_):
                    if ID not in data:
                        data[ID] = {}
                    if "en" not in data.get(ID, {}):
                        data[ID]["en"] = {}
                    data[ID]["en"]["label"] = tgt_text[index]
            pickle.dump(data, open(args["export_filename"], "wb"))
        ids = des_trans[0][i]
        texts = des_trans[1][i]
        if len(ids) > 0:
            for k in tqdm(range(0, len(ids), batch_size)):
                id_ = ids[k : min(k + batch_size, len(ids))]
                texts_ = texts[k : min(k + batch_size, len(texts))]
                tgt_text = translate(texts_, tokenizer, model, device=device)
                for index, ID in enumerate(id_):
                    if ID not in data:
                        data[ID] = {}
                    if "en" not in data[ID]:
                        data[ID]["en"] = {}
                    data[ID]["en"]["description"] = tgt_text[index]
            pickle.dump(data, open(args["export_filename"], "wb"))

    ids = label_trans[0][-1]
    for ID in ids:
        if ID not in data:
            data[ID] = {}
        if "en" not in data[ID]:
            data[ID]["en"] = {}
        data[ID]["en"]["label"] = "Unknown"
    ids = des_trans[0][-1]
    for ID in ids:
        if ID not in data:
            data[ID] = {}
        if "en" not in data[ID]:
            data[ID]["en"] = {}
        data[ID]["en"]["description"] = "Unknown"

    pickle.dump(data, open(args["export_filename"], "wb"))
    original_data = pickle.load(open(file_path, "rb"))
    final_res = merge_dicts(init_dict=original_data, res_dict=data)
    all_ = [
        (
            key,
            value.get("instance", "Unknown"),
            value["en"]["label"],
            value["en"]["description"],
        )
        for key, value in final_res.items()
    ]
    df = pd.DataFrame(all_)
    df.columns = ["id", "instance", "label", "description"]
    # Output the final result
    df.to_csv(os.path.join("data", "Mapping", "Final_result.csv"), index=False)
    return df


def convert_data(data: dict):
    li = []
    for key in data:
        data[key]["id"] = key
        li.append(data[key])
    return li


def get_batch_dfs(li: list, batch=100000):
    dfs = []
    for i in range(0, len(li), batch):
        print(i)
        dfs.append(json_normalize(li[i : i + batch]))
    return dfs


if __name__ == "__main__":
    args = parser.parse_args().__dict__
    lan_list = args.get("lan_list", [])
    if lan_list is None or len(lan_list) == 0:
        lan_list = ["sv", "roa", "de", "zh", "ja", "ar", "ko", "id", "ru", "nl"]
    lan_index = {lan_list[i]: i for i in range(len(lan_list))}

    start_time = time.time()
    print(args["lan_list"])
    _ = translate_mappings(args)
    print(f"This took {time.time() - start_time} seconds")
