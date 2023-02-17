import pandas as pd
import json
import re


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


def convert_pd_ents_to_dict():
    df = pd.read_csv("data/Mapping/Final_result.csv")
    df.shape
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


if __name__ == "__main__":
    convert_pd_ents_to_dict()
