import argparse
from str2bool import str2bool

import re

from sentence_transformers import SentenceTransformer
import json
import torch
from tqdm import tqdm
import numpy as np
import wordninja
import time
import os
from typing import Dict
import sys
import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.getcwd())
from src.utils import typed, seed_everything  # noqa: E402

assert load_dotenv()

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
    with open(os.environ["DATA_PATH"] + "{}/ent2ids".format(dataset), "r") as f:
        entity2id = json.load(f)
    with open(os.environ["DATA_PATH"] + "{}/relation2ids".format(dataset), "r") as f:
        rel2ids = json.load(f)
    if dataset == "Wiki" or use_extra_descs == "_extra":
        with open(os.environ["DATA_PATH"] + f"{dataset}/{entity_text_file}", "r") as f:
            ent2entids = json.load(f)

        with open(
            os.environ["DATA_PATH"] + f"{dataset}/relationdescriptions", "r"
        ) as f:
            relationdescriptions = json.load(f)

        ents_to_ret = {}

        for ent_id in entity2id:
            ents_to_ret[entity2id[ent_id]] = ent2entids[ent_id]

        rels_to_ret = {}
        for rel_id in relationdescriptions:
            rels_to_ret[rel2ids[rel_id]] = get_relation(
                relationdict=relationdescriptions[rel_id],
                entity_text_file=entity_text_file,
            ).strip()
        print(
            f"Subsample: {list(rels_to_ret.keys())[:10]} : {list(rels_to_ret.values())[:10]} "
        )
        return entity2id, ents_to_ret, rel2ids, rels_to_ret
    ents_to_ret = {}
    rels_to_ret = {}
    for ent in entity2id:
        ents_to_ret[entity2id[ent]] = ent
    for rel in rel2ids:
        rels_to_ret[rel2ids[rel]] = rel
    return entity2id, ents_to_ret, rel2ids, rels_to_ret


def construct_rel_word(
    rel: str, dataset: str, use_extra_descs: bool, relation_descriptions: Dict[str, str]
):

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
        return correct_sentence(" ".join(res_)) + (
            " " + relation_descriptions[rel]
            if relation_descriptions is not None
            else ""
        )
    elif dataset == "NELL" and use_extra_descs == "_extra":
        return rel.strip()
    elif dataset == "Wiki":
        return " ".join([x.lower() for x in wordninja.split(rel)])
    else:
        raise NotImplementedError(
            f"Not implemented relation preprocessing for dataset {dataset}."
        )


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
    # print(f"ent : {ent}")
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


def infer_sents_BERT(sents, model, train=False, batch=256):
    # sents_ = [x.split(" ") for x in sents]
    if torch.cuda.is_available() is True:
        model.cuda()
    ret = []
    if batch is not None:
        for i in tqdm(range(0, len(sents), batch)):
            pass
            ret.append(model.encode(sents[i : (i + batch)]))
    else:
        ret.append(model.encode(sents))
    # import ipdb; ipdb.sset_trace()
    if isinstance(ret, list):
        assert all([isinstance(x, np.ndarray) for x in ret])
        ret = np.concatenate(ret, axis=0)
    else:
        assert isinstance(ret, np.ndarray)
    return ret


def main(args):
    # dataset= "NELL"
    seed_everything()
    dataset = args.dataset
    ablation_remove = args.ablation_remove
    use_extra_descs = args.use_extra_descs
    if args.ablation_remove is not None:
        assert dataset == "NELL"
        assert use_extra_descs == ""
        use_extra_descs = f"ablation_remove_{ablation_remove}"

    if dataset == "NELL":
        # import ipdb; ipdb.sset_trace()
        relations = pd.read_csv(
            "src/origin_data/NELL/NELL_relation_descriptions.csv", sep="\t"
        )
        relations = relations.drop_duplicates(subset=["relation"])
        relations = relations.set_index("relation")
        relation_descriptions = relations["description"].to_dict()
    else:
        relation_descriptions = None

    ent2ids, ents_to_ret, rel2ids, rels_to_ret = load_ids(
        dataset, use_extra_descs=use_extra_descs, entity_text_file=args.entity_text_file
    )

    ids2ent2words = {}
    ids2rel2words = {}
    print(f"Constructing and preprocessing entity descriptions for {dataset}")
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
    print(f"Constructing and preprocessing relation descriptions for {dataset}")
    for i, reltext in tqdm(rels_to_ret.items()):
        ids2rel2words[i] = construct_rel_word(
            reltext,
            dataset=dataset,
            use_extra_descs=use_extra_descs,
            relation_descriptions=relation_descriptions,
        )

    print("Training model! ")
    model = SentenceTransformer(
        "paraphrase-MiniLM-L12-v2"
    )  # BERT(glove, workers=2, lang_freq="en")
    entity_sents = sorted(list(ids2ent2words.items()), key=lambda x: x[0])
    rel_sents = sorted(list(ids2rel2words.items()), key=lambda x: x[0])
    sents = entity_sents + rel_sents

    # _ = infer_sents([x[1] for x in sents], model, train=True)

    entity_embeddings = infer_sents_BERT(
        [x[1] for x in entity_sents], model, train=False
    )
    rel_embeddings = infer_sents_BERT([x[1] for x in rel_sents], model, train=False)

    # correct = np.loadtxt("src/origin_data/NELL/entity2vec.BERT")
    # assert np.all(correct[0] == entity_embeddings[0])
    if ablation_remove is not None:
        assert use_extra_descs == f"ablation_remove_{ablation_remove}"
    np.savez(
        os.path.join(os.environ["DATA_PATH"], dataset, "BERT_embed.npz"),
        eM=entity_embeddings,
        rM=rel_embeddings,
    )
    if args.generate_for_new_relations is True:
        tasks2ids = {}
        i = 0
        for task in ["train_tasks.json", "dev_tasks.json", "test_tasks.json"]:
            tasks = json.load(open(os.path.join("data", args.dataset, task)))
            for relation in tasks.keys():
                tasks2ids[relation] = construct_rel_word(
                    relation,
                    dataset=args.dataset,
                    use_extra_descs=args.use_extra_descs,
                    relation_descriptions=relation_descriptions,
                )
        sents = list(tasks2ids.items())

        original_ = [x[0] for x in sents]
        task2ids = dict([(sent, i) for i, sent in enumerate(original_)])
        sents = [x[1] for x in sents]
        sentences = []
        for sent in sents:
            sentences.append(sent.split(" "))
        # s = IndexedList(sentences)
        # vecs = model.infer(s)
        vecs = infer_sents_BERT(sents=sentences, model=model)
        task_embeddings = np.zeros((len(tasks2ids), vecs.shape[1]))
        for i, s_ in tqdm(enumerate(original_)):
            if s_ in tasks2ids:
                task_embeddings[task2ids[s_]] = vecs[i]
            else:
                raise Exception("{} not in _2ids".format(s_))
        np.savetxt(
            os.environ["DATA_PATH"]
            + "{}/tasks2vec.BERT{}".format(dataset, use_extra_descs),
            task_embeddings,
        )
        with open(
            os.environ["DATA_PATH"] + "{}/task2ids".format(args.dataset), "w"
        ) as f:
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

if __name__ == "__main__":
    start_time = time.time()
    args = parser.parse_args()
    # import ipdb; ipdb.sset_trace()
    # with ipdb.launch_ipdb_on_exception():
    main(args)
    print(f"This took {(time.time() - start_time)/60} minutes. ")
