import pickle
import os
from wikidata.client import Client
import json


def main():
    rels = pickle.load(open(os.path.join("data", "Mapping", "relations.p"), "rb"))

    client = Client()
    # P2237 Replaced with 2302 : https://www.wikidata.org/wiki/Property_talk:P2237
    res_2237 = client.get("P2302")
    # res_2237.label["en"]
    rels["P2237"] = {
        "label": res_2237.label["en"],
        "description": res_2237.description["en"],
    }
    rels["P1773"] = {"label": "creator", "description": "of"}

    for rel in list(rels.keys()):
        rels[f"{rel}_inv"] = rels[rel]
        rels[f"{rel}_inv"]["label"] = rels[rel]["label"] + " inverse"

    with open("data/Wiki/relationdescriptions", "w") as f:
        json.dump(rels, f)


if __name__ == "__main__":
    main()
