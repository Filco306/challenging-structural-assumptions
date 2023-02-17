import wordninja
import re
import os
import json
import pandas as pd
import sys

sys.path.append(os.getcwd())
from scripts.generate_sif_embeddings import (  # noqa: E402
    construct_ent_word,
    correct_sentence,
)
from scripts.generate_sif_embeddings import construct_rel_word  # noqa: E402


def main():
    df = pd.read_csv("data/NELL/NELL.08m.1115.esv.csv", sep="\t", error_bad_lines=False)
    ent2ids = json.load(open("data/NELL/ent2ids", "r"))
    rel2ids = json.load(open("data/NELL/relation2ids", "r"))
    df["Categories for Entity"].replace("concept:", "", regex=True, inplace=True)
    df["Categories for Value"].replace("concept:", "", regex=True, inplace=True)
    entities = df[["Entity", "Categories for Entity"]].drop_duplicates()
    entities.columns = ["ent", "desc"]
    entities2 = df[["Value", "Categories for Value"]].drop_duplicates()
    entities2.columns = ["ent", "desc"]
    ents = pd.concat([entities, entities2], axis=0, ignore_index=True).drop_duplicates()
    ents["desc"] = ents["desc"].astype("str")
    ents_NELL_data = pd.DataFrame(list(ent2ids.keys()), columns=["ent"])
    res = ents.merge(ents_NELL_data, how="right", on="ent")
    res["desc"] = res["desc"].fillna("").replace("(nan)|(NaN)", "", regex=True)
    res["final"] = (
        res["ent"].apply(lambda x: construct_ent_word(x, "NELL"))
        + " "
        + res["desc"].apply(lambda x: correct_sentence(" ".join(wordninja.split(x))))
    ).apply(lambda x: re.sub(" +", " ", " ".join(list(set(x.split(" "))))).strip())
    out = dict(zip(res["ent"], res["final"]))
    with open(os.path.join("data", "NELL", "enttext2entids"), "w") as f:
        json.dump(out, f)

    relationsdescriptions_NELL = {}
    for rel in rel2ids:
        relationsdescriptions_NELL[rel] = {
            "label": construct_rel_word(rel, "NELL"),
            "description": "",
        }
    with open(os.path.join("data", "NELL", "relationdescriptions"), "w") as f:
        json.dump(relationsdescriptions_NELL, f)
