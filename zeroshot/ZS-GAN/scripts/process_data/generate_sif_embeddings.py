import json
import wordninja
import re
import argparse
import gensim.downloader as api
from fse.models import uSIF
from fse import IndexedList
import numpy as np
import os
from str2bool import str2bool
from typing import Dict, Callable
import sys
import time
from tqdm import tqdm
import logging
import pandas as pd
sys.path.append(os.getcwd())
from src.utils import typed, seed_everything  # noqa: E402

INCORRECT_VERSIONS = {
    "knott s": "knotts",
    "s often": "soften",
    "con funded": "confunded",
    "austro naut s": "austronauts",
    "visual iz able": "visualizable",
    "001": "",
    "n 2": "two",
    "steven der o uni an": "steven derounian",
    "zac arias moussa oui": "zacarias moussaoui",
}
CORRECTION = "(" + ")|(".join(list(INCORRECT_VERSIONS.keys())) + ")"


def load_w2vec(vecs: str = "word2vec-google-news-300"):
    model = api.load(vecs)
    return model


@typed()
def get_relation(relationdict: Dict[str, str], entity_text_file: str) -> str:
    if entity_text_file == "enttext2id_ablation_no_description":
        return relationdict["label"]
    elif entity_text_file == "enttext2id_ablation_no_label":
        return relationdict["description"]
    elif entity_text_file in ["enttext2id_ablation_no_instance", "enttext2entids"]:
        return relationdict["label"] + " " + relationdict["description"]
    else:
        raise KeyError(
            f"entity_text_file {entity_text_file} was unexpected. Returning error. "
        )


@typed()
def load_ids(
    dataset: str, use_extra_descs: str, entity_text_file: str = "enttext2entids"
):
    with open("src/origin_data/{}/entity2id".format(dataset), "r") as f:
        ent2ids = json.load(f)
    with open("src/origin_data/{}/relation2ids".format(dataset), "r") as f:
        rel2ids = json.load(f)
    if dataset == "Wiki" or use_extra_descs == "_extra":
        with open(f"src/origin_data/{dataset}/{entity_text_file}", "r") as f:
            ent2entids = json.load(f)

        with open(f"src/origin_data/{dataset}/relationdescriptions", "r") as f:
            relationdescriptions = json.load(f)

        ents_to_ret = {}
        for ent_id in ent2ids:
            ents_to_ret[ent2ids[ent_id]] = ent2entids[ent_id]

        rels_to_ret = {}
        for rel_id in relationdescriptions:
            if rel_id in rel2ids:
                rels_to_ret[rel2ids[rel_id]] = get_relation(
                    relationdict=relationdescriptions[rel_id],
                    entity_text_file=entity_text_file,
                ).strip()
            else:
                logging.warn(f"Id {rel_id} not found in rel2ids. ")

        logging.info(
            f"Subsample: {list(rels_to_ret.keys())[:10]} : {list(rels_to_ret.values())[:10]} "
        )
        return ent2ids, ents_to_ret, rel2ids, rels_to_ret
    ents_to_ret = {}
    rels_to_ret = {}
    for ent in ent2ids:
        ents_to_ret[ent2ids[ent]] = ent
    for rel in rel2ids:
        rels_to_ret[rel2ids[rel]] = rel
    return ent2ids, ents_to_ret, rel2ids, rels_to_ret


def correct_sentence(sentence):
    sent_ = sentence
    while re.search(CORRECTION, sent_) is not None:
        replace_ = re.search(CORRECTION, sent_).group(0)
        if re.search("n\d+", replace_) is not None:  # noqa: W605
            sent_ = re.sub(replace_, INCORRECT_VERSIONS["n\d+"], sent_)  # noqa: W605
        else:
            sent_ = re.sub(replace_, INCORRECT_VERSIONS[replace_], sent_)
    return sent_


@typed()
def construct_ent_word(
    ent: str, dataset: str, use_extra_descs, ablation_remove: str = None
):
    # logging.info(f"ent : {ent}")
    if dataset == "NELL" and use_extra_descs != "_extra":
        if re.search("\d+-\d+", ent) is not None:  # noqa: W605
            return correct_sentence("score " + re.sub("-", " to ", ent))
        elif ent.isnumeric():
            return correct_sentence("cardinality " + ent)
        else:
            type_of, entity = ent.split(":")[1:]
        if type_of == "nonneginteger":
            if re.search("n(\d+)", entity) is None:  # noqa: W605
                return correct_sentence(
                    "positive integer " + " ".join(wordninja.split(entity))
                )

            res = correct_sentence(
                " ".join(
                    [
                        "positive integer",
                        re.search("n(\d+)", entity).group(1),  # noqa: W605
                    ]  # noqa: W605
                )
            )
            return res
        elif ablation_remove is not None:
            assert ablation_remove in ["type", "label"]
            if ablation_remove == "type":
                return re.sub("_", " ", entity)
            elif ablation_remove == "label":
                return " ".join(wordninja.split(type_of))
        else:
            assert ablation_remove is None
            return correct_sentence(
                " ".join(wordninja.split(type_of) + [re.sub("_", " ", entity)])
            )
    elif dataset == "Wiki":
        assert ablation_remove is None
        return " ".join([x.lower() for x in wordninja.split(ent)])
    elif use_extra_descs == "_extra":
        assert ablation_remove is None
        return ent
    else:
        raise NotImplementedError(
            f"Not implemented entity preprocessing for dataset {dataset}."
        )


def construct_rel_word(rel: str, dataset: str, use_extra_descs: bool, relation_descriptions : Dict[str, str]):
    
    if dataset == "NELL" and use_extra_descs != "_extra":
        # if ablation_remove is None:
        rel_ = rel.split(":")[1:]

        assert isinstance(rel_, list) and len(rel_) == 1
        rel_ = rel_[0]
        res_ = wordninja.split(rel_)
        if " ".join(res_[-2:]) == "in v":
            # import ipdb; ipdb.sset_trace()
            res_ = res_[:-2]
            res_ = ["inverse"] + res_
        # import ipdb; ipdb.sset_trace()
        return correct_sentence(" ".join(res_)) + (" " + relation_descriptions[rel] if relation_descriptions is not None else "")
    elif dataset == "NELL" and use_extra_descs == "_extra":
        return rel.strip()
    elif dataset == "Wiki":
        return " ".join([x.lower() for x in wordninja.split(rel)])
    else:
        raise NotImplementedError(
            f"Not implemented relation preprocessing for dataset {dataset}."
        )


def convert_triplets(path_graph: str, dataset: str):
    with open(path_graph, "r") as f:
        sentences = [x.split("\t") for x in f.read().split("\n")[:-1]]
    ret = []
    for i in range(len(sentences)):
        head, rel, tail = sentences[i]
        ret.append(
            " ".join(
                (
                    construct_ent_word(head, dataset=dataset),
                    construct_rel_word(rel, dataset=dataset, relation_descriptions=relation_descriptions),
                    construct_ent_word(tail, dataset=dataset),
                )
            )
        )
    return ret


@typed()
def fix_embeddings(_2ids: Dict[str, int], construct_word_func: Callable):
    pass


def infer_sents(sents, model, train=False):
    sents_ = [x.split(" ") for x in sents]
    s_ = IndexedList(sents_)
    if train is True:
        model.train(s_)
    return model.infer(s_)


def main(args):
    # dataset= "NELL"
    seed_everything()
    dataset = args.dataset
    ablation_remove = args.ablation_remove
    use_extra_descs = args.use_extra_descs
    if args.ablation_remove is not None:
        assert use_extra_descs == ""
        use_extra_descs = f"ablation_remove_{ablation_remove}"

    ent2ids, ents_to_ret, rel2ids, rels_to_ret = load_ids(
        dataset, use_extra_descs=use_extra_descs, entity_text_file=args.entity_text_file
    )
    

    if dataset == "NELL":
        # import ipdb; ipdb.sset_trace()
        relations = pd.read_csv("src/origin_data/NELL/NELL_relation_descriptions.csv", sep = "\t")
        relations = relations.drop_duplicates(subset=["relation"])
        relations = relations.set_index("relation")
        relation_descriptions = relations["description"].to_dict()
    else:
        relation_descriptions = None
    
    ids2ent2words = {}
    ids2rel2words = {}
    
    logging.info(f"Constructing and preprocessing relation descriptions for {dataset}")
    # import ipdb; ipdb.sset_trace()
    for i, reltext in tqdm(rels_to_ret.items()):
        ids2rel2words[i] = construct_rel_word(
            reltext, dataset=dataset, use_extra_descs=use_extra_descs, relation_descriptions=relation_descriptions
        )
        # import ipdb; ipdb.sset_trace()

    logging.info(f"Constructing and preprocessing entity descriptions for {dataset}")
    for i, enttext in tqdm(ents_to_ret.items()):
        if (
            construct_ent_word(
                ents_to_ret[i],
                dataset=dataset,
                use_extra_descs=use_extra_descs,
                ablation_remove=ablation_remove,
            )
            in ids2ent2words
        ):
            raise Exception(f"{i} already accounted for : {enttext}")
        ids2ent2words[i] = construct_ent_word(
            enttext,
            dataset=dataset,
            use_extra_descs=use_extra_descs,
            ablation_remove=ablation_remove,
        )
        # import ipdb; ipdb.sset_trace()

    # glove = load_w2vec("glove-wiki-gigaword-100")
    glove = load_w2vec(args.w2vec_path)
    logging.info("Training model! ")
    model = uSIF(glove, workers=2, lang_freq="en")
    entity_sents = sorted(list(ids2ent2words.items()), key=lambda x: x[0])
    rel_sents = sorted(list(ids2rel2words.items()), key=lambda x: x[0])
    sents = entity_sents + rel_sents

    _ = infer_sents([x[1] for x in sents], model, train=True)

    entity_embeddings = infer_sents([x[1] for x in entity_sents], model, train=False)
    rel_embeddings = infer_sents([x[1] for x in rel_sents], model, train=False)

    # correct = np.loadtxt("src/origin_data/NELL/entity2vec.uSIF")
    # assert np.all(correct[0] == entity_embeddings[0])
    if ablation_remove is not None:
        assert use_extra_descs == f"ablation_remove_{ablation_remove}"

    if args.save_npz is True:
        np.savez(
            os.path.join("src/origin_data/", dataset, "SIF_embed.npz"),
            eM=entity_embeddings,
            rM=rel_embeddings,
        )
    else:
        np.savetxt(
            "src/origin_data/{}/entity2vec.uSIF{}".format(dataset, use_extra_descs),
            entity_embeddings,
        )
        np.savetxt(
            "src/origin_data/{}/relation2vec.uSIF{}".format(dataset, use_extra_descs),
            rel_embeddings,
        )
    if args.generate_for_new_relations is True:
        tasks2ids = {}
        i = 0
        for task in ["train_tasks.json", "dev_tasks.json", "test_tasks.json"]:
            tasks = json.load(open(os.path.join("data", args.dataset, task)))
            for relation in tasks.keys():
                tasks2ids[relation] = construct_rel_word(
                    relation, dataset=args.dataset, use_extra_descs=args.use_extra_descs, relation_descriptions=relation_descriptions
                )
        sents = list(tasks2ids.items())

        original_ = [x[0] for x in sents]
        task2ids = dict([(sent, i) for i, sent in enumerate(original_)])
        sents = [x[1] for x in sents]
        sentences = []
        for sent in sents:
            sentences.append(sent.split(" "))
        s = IndexedList(sentences)
        vecs = model.infer(s)
        task_embeddings = np.zeros((len(tasks2ids), vecs.shape[1]))
        for i, s_ in tqdm(enumerate(original_)):
            if s_ in tasks2ids:
                task_embeddings[task2ids[s_]] = vecs[i]
            else:
                raise Exception("{} not in _2ids".format(s_))
        np.savetxt(
            "src/origin_data/{}/tasks2vec.uSIF{}".format(dataset, use_extra_descs),
            task_embeddings,
        )
        with open("src/origin_data/{}/task2ids".format(args.dataset), "w") as f:
            json.dump(task2ids, f)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--w2vec_path",
    type=str,
    default="glove-wiki-gigaword-100",
    help="""Name of w2vec embeddings. Can be 'glove-wiki-gigaword-100'
            or 'word2vec-google-news-300' as two examples. """,
)
parser.add_argument("--dataset", type=str, choices=["NELL", "Wiki"], default="NELL")
parser.add_argument("--generate_for_new_relations", type=str2bool, default=False)
parser.add_argument("--save_npz", type=str2bool, default=True)
parser.add_argument(
    "--use_extra_descs",
    type=str,
    default="",
    help="Whether to include additional descriptions or not for the NELL dataset. ",
)
parser.add_argument(
    "--entity_text_file",
    type=str,
    default="enttext2entids",
    help="Argument used to facilitate generation of ablation study vectors. ",
)
parser.add_argument(
    "--ablation_remove",
    type=str,
    default=None,
    choices=[None, "type", "label"],
    help="Argument used to facilitate generation of ablation study vectors. ",
)
# import ipdb
# with ipdb.launch_ipdb_on_exception():
if __name__ == "__main__":
    start_time = time.time()
    args = parser.parse_args()

    main(args)
    logging.info(f"This took {(time.time() - start_time)/60} minutes. ")
