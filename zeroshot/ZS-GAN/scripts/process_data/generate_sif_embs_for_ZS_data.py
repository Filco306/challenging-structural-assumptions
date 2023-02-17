import os
import json
import logging
from tqdm import tqdm
from wikidata.client import Client
from googletrans import Translator
import urllib


def fix_relationdescriptions_wiki():
    dataset = "Wiki"

    with open(os.path.join("src/origin_data", dataset, "relation2ids"), "r") as f:
        new_rel2id = json.load(f)

    with open(
        os.path.join("src/origin_data", dataset, "relationdescriptions"), "r"
    ) as f:
        old_rel_descs = json.load(f)

    olds_ = set(list(old_rel_descs.keys()))

    wikiclient = Client()

    tr = Translator()

    not_found = []
    for relation in tqdm([x for x in set(list(new_rel2id.keys())) if x not in olds_]):
        new_rel = wikiclient.get(relation)
        try:
            old_rel_descs[relation] = {
                "label": new_rel.label["en"]
                if "en" in new_rel.label
                else tr.translate(new_rel.label[list(new_rel.label.keys())[0]]).text,
                "description": new_rel.description["en"]
                if "en" in new_rel.description
                else tr.translate(
                    new_rel.description[list(new_rel.description.keys())[0]]
                ).text,
            }
        except urllib.error.HTTPError as e:  # noqa: F841
            logging.info(f"{relation} not found")
            not_found.append(relation)
            old_rel_descs[relation] = {"label": "unknown", "description": "unknown"}
    with open(
        os.path.join("src/origin_data", dataset, "relationdescriptions"), "w"
    ) as f:
        json.dump(old_rel_descs, f)


if __name__ == "__main__":
    fix_relationdescriptions_wiki()
