import typesentry
import numpy as np
import json
import logging
import torch
from torch.autograd import Variable
from collections import defaultdict
from typing import Dict
from tqdm import tqdm
import os



def load_embed(dataset: str, embed_model: str):
    # gen symbol2id, with embedding
    symbol_id = {}
    rel2id = json.load(
        open("data/" + dataset + "/relation2ids")
    )  # relation2id contains inverse rel
    ent2id = json.load(open("data/" + dataset + "/ent2ids"))

    logging.info("LOADING PRE-TRAINED EMBEDDING")
    if embed_model in [
        "DistMult",
        "TransE",
        "ComplEx",
        "RESCAL",
        "uSIF",
    ]:
        ent_embed = np.loadtxt("data/" + dataset + "/entity2vec." + embed_model)
        rel_embed = np.loadtxt(
            "data/" + dataset + "/relation2vec." + embed_model
        )  # contain inverse edge

        if embed_model == "ComplEx":
            # normalize the complex embeddings
            ent_mean = np.mean(ent_embed, axis=1, keepdims=True)
            ent_std = np.std(ent_embed, axis=1, keepdims=True)
            rel_mean = np.mean(rel_embed, axis=1, keepdims=True)
            rel_std = np.std(rel_embed, axis=1, keepdims=True)
            eps = 1e-3
            ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
            rel_embed = (rel_embed - rel_mean) / (rel_std + eps)

        assert ent_embed.shape[0] == len(ent2id.keys())
        assert rel_embed.shape[0] == len(rel2id.keys()), "{} != {}".format(
            rel_embed.shape[0], len(rel2id.keys())
        )

        i = 0
        embeddings = []
        for key in rel2id.keys():
            if key not in ["", "OOV"]:
                symbol_id[key] = i
                i += 1
                embeddings.append(list(rel_embed[rel2id[key], :]))

        for key in ent2id.keys():
            if key not in ["", "OOV"]:
                symbol_id[key] = i
                i += 1
                embeddings.append(list(ent_embed[ent2id[key], :]))

        symbol_id["PAD"] = i
        embeddings.append(list(np.zeros((rel_embed.shape[1],))))
        embeddings = np.array(embeddings)
        assert embeddings.shape[0] == len(symbol_id.keys())

        symbol2id = symbol_id
        symbol2vec = embeddings
        return symbol2id, symbol2vec



def load_symbol2id(dataset: str):
    # gen symbol2id, without embedding
    symbol_id = {}
    rel2id = json.load(open("data/" + dataset + "/relation2ids"))
    ent2id = json.load(open("data/" + dataset + "/ent2ids"))
    i = 0
    # rel and ent combine together
    for key in rel2id.keys():
        if key not in ["", "OOV"]:
            symbol_id[key] = i
            i += 1

    for key in ent2id.keys():
        if key not in ["", "OOV"]:
            symbol_id[key] = i
            i += 1

    symbol_id["PAD"] = i
    symbol2id = symbol_id
    symbol2vec = None
    return symbol2id, symbol2vec



def get_meta(left, right, connections: np.ndarray, e1_degrees, device):
    # Here we get the meta information
    left_connections = Variable(
        torch.LongTensor(np.stack([connections[_, :, :] for _ in left], axis=0))
    ).to(device)
    left_degrees = Variable(torch.FloatTensor([e1_degrees[_] for _ in left])).to(device)
    right_connections = Variable(
        torch.LongTensor(np.stack([connections[_, :, :] for _ in right], axis=0))
    ).to(device)
    right_degrees = Variable(torch.FloatTensor([e1_degrees[_] for _ in right])).to(
        device
    )
    return (left_connections, left_degrees, right_connections, right_degrees)



def build_connection(
    ent2id: Dict[str, int],
    symbol2id: Dict[str, int],
    dataset: str,
    num_ents: int,
    pad_id: int,
    max_: int = 100,
):
    # We take the relations into account here,
    connections = (np.ones((num_ents, max_, 2)) * pad_id).astype(int)
    e1_rele2 = defaultdict(list)
    e1_degrees = defaultdict(int)
    with open("data/" + dataset + "/path_graph") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            e1, rel, e2 = line.rstrip().split()
            e1_rele2[e1].append((symbol2id[rel], symbol2id[e2]))  # 1-n
            e1_rele2[e2].append((symbol2id[rel + "_inv"], symbol2id[e1]))  # n-1

    degrees = {}
    for ent, id_ in ent2id.items():
        neighbors = e1_rele2[ent]
        if len(neighbors) > max_:
            neighbors = neighbors[:max_]
        degrees[ent] = len(neighbors)
        e1_degrees[id_] = len(neighbors)  # add one for self conn
        for idx, _ in enumerate(neighbors):
            connections[id_, idx, 0] = _[0]  # rel
            connections[id_, idx, 1] = _[1]  # tail
    return connections, e1_rele2, e1_degrees



def load_tasks(
    dataset: str = "NELL",
    load_train: bool = False,
    load_dev: bool = True,
    load_test: bool = True,
):
    assert any([load_dev, load_test]), ""
    train_ = (
        json.load(open(os.path.join("data", "{}".format(dataset), "train_tasks.json")))
        if load_train is True
        else {}
    )
    dev_ = (
        json.load(open(os.path.join("data", "{}".format(dataset), "dev_tasks.json")))
        if load_dev is True
        else {}
    )
    test_ = (
        json.load(open(os.path.join("data/{}/test_tasks.json".format(dataset))))
        if load_test is True
        else {}
    )
    print("train_ : ", len(train_))
    print("dev_ : ", len(dev_))
    print("test_ : ", len(test_))
    return {**train_, **dev_, **test_}


class ArgMocker:
    
    def __init__(self, args: dict):
        self.__dict__ = args

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value



def getX2id(trainer, xx: str = "symbol2id"):
    return {v: k for k, v in getattr(trainer, xx).items()}
