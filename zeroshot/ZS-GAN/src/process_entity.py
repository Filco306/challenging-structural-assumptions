# -*- coding: utf-8 -*-
import json
from collections import defaultdict
import logging


def type_to_entity():
    """
    build candiate tail entities for every relation
    """
    # calculate node degrees

    ent2ids = json.load(open("../ent2ids"))

    all_entities = ent2ids.keys()

    type2ents = defaultdict(set)
    entity_exception = list()

    for ent in all_entities:
        if len(ent.split(":")) != 3:
            logging.info(ent)
            entity_exception.append(ent)
        try:
            type_ = ent.split(":")[1]
            type2ents[type_].add(ent)
        except Exception as e:  # noqa: F841
            continue

    for k, v in type2ents.items():
        type2ents[k] = list(v)

    json.dump(type2ents, open("../type2ents.json", "w"))

    logging.info("Entity Exception:", len(entity_exception))

    logging.info("type2sent:", len(type2ents))

    for k, v in type2ents.items():
        logging.info(k, "        ", len(v))


def process_tasks(train_tasks, mode):
    type2ents = json.load(open("../type2ents.json"))
    type2rela = json.load(open("../e_type2rela.json"))
    rela2relas = dict()

    wrong = set()
    for rela, triples in train_tasks.items():
        e1_types = set()
        e2_types = set()
        related_rela = set()
        e1_rela = set()
        e2_rela = set()
        for triple in triples:
            e1, e2 = triple[0], triple[2]
            e1_t = e1.split(":")[1]
            e2_t = e2.split(":")[1]

            if e1_t in type2ents.keys():
                e1_types.add(e1_t)
            else:
                wrong.add(e1)

            if e2_t in type2ents.keys():
                e2_types.add(e2_t)
            else:
                wrong.add(e2)

        for e1_t in e1_types:
            if e1_t in type2rela.keys():
                for item in type2rela[e1_t]:
                    related_rela.add(item)
                    e1_rela.add(item)

        for e2_t in e2_types:
            if e2_t in type2rela.keys():
                for item in type2rela[e2_t]:
                    related_rela.add(item)
                    e2_rela.add(item)

        rela2relas[rela] = list(related_rela)
        logging.info("RELATION: ", rela, len(related_rela), len(e1_rela), len(e2_rela))

    logging.info("WRONG: ", len(wrong))

    json.dump(rela2relas, open("../" + mode + "_rela2relas.json", "w"))


def e1e2_type2rela(pretrain_tasks):

    e1e2_type2rela = defaultdict(set)

    for rela, triples in pretrain_tasks.items():
        for triple in triples:
            e1 = triple[0]
            e2 = triple[2]
            if len(e1.split(":")) != 3 or len(e2.split(":")) != 3:
                continue
            e1_type = e1.split(":")[1]
            e2_type = e2.split(":")[1]
            e1e2_type2rela[e1_type + "#" + e2_type].add(rela)
            e1e2_type2rela[e2_type + "#" + e1_type].add(rela)

    logging.info("length: ", len(e1e2_type2rela))
    count2 = 0
    relations = list()
    for k, v in e1e2_type2rela.items():
        relations += v
        if len(v) > 1:
            count2 += 1

    logging.info("larger than 2: ", count2)
    logging.info(len(set(relations)))

    for k, v in e1e2_type2rela.items():
        e1e2_type2rela[k] = list(v)
    json.dump(e1e2_type2rela, open("../e1e2_type2rela.json", "w"))


def e_type2rela(pretrain_tasks):

    e_type2rela = defaultdict(set)

    for rela, triples in pretrain_tasks.items():
        if len(triples) < 500:
            continue
        for triple in triples:
            e1 = triple[0]
            e2 = triple[2]
            if len(e1.split(":")) != 3 or len(e2.split(":")) != 3:
                continue
            e1_type = e1.split(":")[1]
            e2_type = e2.split(":")[1]
            e_type2rela[e1_type].add(rela)
            e_type2rela[e2_type].add(rela)

    logging.info("length: ", len(e_type2rela))

    count = 0
    for k, v in e_type2rela.items():
        if len(v) > 28:
            count += 1
            del e_type2rela[k]
            continue
        e_type2rela[k] = list(v)
    logging.info("Common Entity: ", count)
    json.dump(e_type2rela, open("../e_type2rela.json", "w"))
