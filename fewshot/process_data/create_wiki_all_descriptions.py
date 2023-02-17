import os
import pickle
import re
import pandas as pd
import json
import sys

sys.path.append(os.getcwd())
from scripts.process_data.translate_mappings import (
    merge_dicts,
    translate_mappings,
)
import argparse
from str2bool import str2bool

"""
    Script used to generate 
"""


def create_wiki_all_descriptions(
    path_total_entity_translated: str = None, dump_file: bool = True
):
    if path_total_entity_translated == "data/Wiki/wiki_all_descriptions":
        with open(path_total_entity_translated, "r") as f:
            final = json.load(f)
    elif path_total_entity_translated is None:
        data = pickle.load(open("data/Mapping/ALMOSTFINALVERSION.p", "rb"))
        swedish_data = pickle.load(open("data/Mapping/swedish_translated.p", "rb"))
        final = merge_dicts(data, swedish_data)
    elif "final_result.csv" in path_total_entity_translated.lower():
        final = pd.read_csv(path_total_entity_translated)
        final = final.set_index("id").to_dict(orient="index")
    elif path_total_entity_translated == "generate":
        final = translate_mappings(
            {
                "file_path": "data/Mapping/total_entity2.p",
                "batch_size": 60,
                "export_filename": "data/Mapping/total_entity_translated.p",
                "lan_list": [
                    "sv",
                    "roa",
                    "de",
                    "zh",
                    "ja",
                    "ar",
                    "ko",
                    "id",
                    "ru",
                    "nl",
                ],
            }
        )
        final = final.set_index("id").to_dict(orient="index")
    else:
        raise ValueError(
            f"Unexpected value of path_total_entity_translated : {path_total_entity_translated}"
        )
    if dump_file is True:
        with open(os.path.join("data", "Wiki", "wiki_all_descriptions"), "w") as f:
            json.dump(final, f)
    return final


def generate_texts_for_inference(final=None, dump_file: bool = False):
    if final is None:
        with open(os.path.join("data", "Wiki", "wiki_all_descriptions"), "r") as f:
            final = json.load(f)

    all_ = [
        (
            key,
            value.get("instance", "Unknown"),
            value["en"]["label"],
            value["en"]["description"],
        )
        for key, value in final.items()
    ]
    df = pd.DataFrame(all_)
    df.columns = ["id", "instance", "label", "description"]
    with open("data/Wiki/ent2ids") as f:
        ent2id = json.load(f)
    all_ents = set(list(ent2id))
    all_ids = set(df["id"].unique().tolist())  # Take all ids from
    ents_and_ids = pd.DataFrame(
        [(k, v) for k, v in ent2id.items()], columns=["id", "index"]
    )
    ents_and_ids
    all_ids_ = []
    for id_ in all_ents:
        if id_ not in all_ids:
            all_ids_.append(id_)
    df_deleted = pd.DataFrame(pd.Series(all_ids_))
    for col in ["instance", "label", "description"]:
        df_deleted[col] = "unknown"
    df_deleted.rename({0: "id"}, axis=1, inplace=True)
    df = pd.concat([df, df_deleted], axis=0, ignore_index=True).merge(
        ents_and_ids, on="id"
    )
    assert df.shape[0] == 4838243
    df["text"] = df["instance"] + " " + df["label"] + " " + df["description"]
    df["ablation_no_description"] = df["instance"] + " " + df["label"]
    df["ablation_no_label"] = df["instance"] + " " + df["description"]
    df["ablation_no_instance"] = df["label"] + " " + df["description"]
    if dump_file is True:
        df[
            [
                "id",
                "index",
                "text",
                "ablation_no_description",
                "ablation_no_label",
                "ablation_no_instance",
            ]
        ].to_csv(os.path.join("data", "Mapping", "Final_result.csv"), index=False)
    return df


def clean(df):
    df["text"] = df["text"].str.lower().replace("(unknown ?)+", "unknown ", regex=True)
    df["ablation_no_description"] = (
        df["ablation_no_description"]
        .str.lower()
        .replace("(unknown ?)+", "unknown ", regex=True)
    )
    df["ablation_no_label"] = (
        df["ablation_no_label"]
        .str.lower()
        .replace("(unknown ?)+", "unknown ", regex=True)
    )
    df["ablation_no_instance"] = (
        df["ablation_no_instance"]
        .str.lower()
        .replace("(unknown ?)+", "unknown ", regex=True)
    )
    return df


def convert_pd_ents_to_dict(df=None):
    if df is None:
        df = pd.read_csv("data/Mapping/Final_result.csv")
    df = clean(df)
    enttext2id = dict(zip(df["id"], df["text"]))
    enttext2id_ablation_no_description = dict(
        zip(df["id"], df["ablation_no_description"])
    )
    enttext2id_ablation_no_label = dict(zip(df["id"], df["ablation_no_label"]))
    enttext2id_ablation_no_instance = dict(zip(df["id"], df["ablation_no_instance"]))

    with open("data/Wiki/enttext2entids", "w") as f:
        json.dump(enttext2id, f)
    with open("data/Wiki/enttext2id_ablation_no_description", "w") as f:
        json.dump(enttext2id_ablation_no_description, f)
    with open("data/Wiki/enttext2id_ablation_no_label", "w") as f:
        json.dump(enttext2id_ablation_no_label, f)
    with open("data/Wiki/enttext2id_ablation_no_instance", "w") as f:
        json.dump(enttext2id_ablation_no_instance, f)


def clean_rels():
    with open("data/Wiki/relationdescriptions", "r") as f:
        relationdescriptions = json.load(f)
    to_delete = []
    for key in relationdescriptions:
        if "_inv_inv" in key:
            to_delete.append(key)
        relationdescriptions[key]["label"] = re.sub(
            "( inverse)+", " inverse", relationdescriptions[key]["label"]
        )
        if "_inv" not in key:
            relationdescriptions[key]["label"] = re.sub(
                "( inverse)+", "", relationdescriptions[key]["label"]
            )
    for key in to_delete:
        del relationdescriptions[key]

    with open("data/Wiki/relationdescriptions", "w") as f:
        json.dump(relationdescriptions, f)


def generate_text_necessary(args):
    final = create_wiki_all_descriptions(
        path_total_entity_translated=args.path_total_entity_translated,
        dump_file=args.dump_file,
    )
    df = generate_texts_for_inference(final=final, dump_file=args.dump_file)
    convert_pd_ents_to_dict(df=df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_total_entity_translated",
        type=str,
        default="data/Wiki/wiki_all_descriptions",
    )
    parser.add_argument("--dump_file", type=str2bool, default=False)
    args = parser.parse_args()
    generate_text_necessary(args)
