import logging
import numpy as np
import torch
from torch.autograd import Variable
from typing import Optional, Dict, List, Union, Tuple
import pandas as pd
import json
from utils import get_meta, load_tasks



def build_evaluation_set(
    trainer,
    triple: Union[Tuple[str, str, str], List[str]],
    candidates: List[str],
    meta: bool,
    symbol2id: Optional[Dict[str, int]],
    ent2id: Optional[Dict[str, int]] = None,
):
    true = triple[2]
    query_pairs = []
    query_pairs.append(
        (symbol2id[triple[0]], symbol2id[triple[2]])
        if symbol2id is not None
        else (triple[0], triple[1])
    )
    if meta:
        query_left = []
        query_right = []
        query_left.append(ent2id[triple[0]] if ent2id is not None else triple[0])
        query_right.append(ent2id[triple[2]] if ent2id is not None else triple[2])
    else:
        query_left, query_right = None, None
    for ent in candidates:
        if (ent not in trainer.e1rel_e2[triple[0] + triple[1]]) and ent != true:
            query_pairs.append(
                [symbol2id[triple[0]], symbol2id[ent]]
                if symbol2id is not None
                else [triple[0], ent]
            )
            if meta:
                query_left.append(trainer.ent2id[triple[0]])
                query_right.append(trainer.ent2id[ent])
    return query_pairs, query_left, query_right



def evaluate(
    trainer,
    mode: str = "dev",
    meta: bool = False,
    save_and_return_scores: bool = False,
    remember: bool = False,
    sample_in_eval: float = 1.0,
):
    assert mode in ["testings", "dev", "test", "train", "all"]
    assert meta == trainer.meta, "{} != {}".format(meta, trainer.meta)
    trainer.Matcher.eval()

    symbol2id = trainer.symbol2id
    ent2id = trainer.ent2id
    few = trainer.few
    if save_and_return_scores is True:
        ret = {}
    logging.info("EVALUATING ON %s DATA" % mode.upper())
    load_train = mode in ["train", "all"]
    load_dev = mode in ["testings", "dev", "all"]
    load_test = mode in ["testings", "test", "all"]

    test_tasks = load_tasks(
        dataset=trainer.dataset,
        load_train=load_train,
        load_test=load_test,
        load_dev=load_dev,
    )

    task2ids = (
        json.load(open("data/{}/task2ids".format(trainer.dataset), "r"))
        if trainer.use_relation_embeddings is True
        else None
    )
    task_rel_embeddings = (
        np.loadtxt("data/{}/tasks2vec.{}".format(trainer.dataset, trainer.embed_model))
        if trainer.use_relation_embeddings is True
        else None
    )
    if task2ids is None:
        assert task_rel_embeddings is None
    else:

        assert len(task2ids) == len(task_rel_embeddings)
    rel2candidates = trainer.rel2candidates

    hits10 = []
    hits5 = []
    hits1 = []
    mrr = []
    for query_ in test_tasks.keys():
        if save_and_return_scores is True:
            ret[query_] = {}
            ret[query_]["queries"] = []
        hits10_ = []
        hits5_ = []
        hits1_ = []
        mrr_ = []
        candidates = rel2candidates[query_]
        support_triples = test_tasks[query_][:few]
        support_pairs = [
            [symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples
        ]

        if meta:
            support_left = [trainer.ent2id[triple[0]] for triple in support_triples]
            support_right = [trainer.ent2id[triple[2]] for triple in support_triples]
            support_meta = get_meta(
                support_left,
                support_right,
                connections=trainer.connections,
                e1_degrees=trainer.e1_degrees,
                device=trainer.device,
            )

        support = Variable(torch.LongTensor(support_pairs)).to(trainer.device)
        if save_and_return_scores is True:
            ret[query_]["support"] = support.cpu().numpy()
        for triple in test_tasks[query_][few:]:

            query_pairs, query_left, query_right = build_evaluation_set(
                trainer,
                triple=triple,
                candidates=candidates
                if sample_in_eval >= 1.0 and mode == "dev"
                else np.random.choice(
                    candidates, size=int(sample_in_eval * len(candidates))
                ).tolist(),
                meta=meta,
                symbol2id=symbol2id,
                ent2id=ent2id,
            )
            query = Variable(torch.LongTensor(query_pairs)).to(trainer.device)

            if meta:
                query_meta = get_meta(
                    query_left,
                    query_right,
                    connections=trainer.connections,
                    e1_degrees=trainer.e1_degrees,
                    device=trainer.device,
                )
            task_rel_emb = (
                torch.Tensor(task_rel_embeddings[task2ids[query_]])
                if task2ids is not None
                else None
            )
            scores, _ = trainer.Matcher(
                support,
                query,
                None,
                isEval=True,
                support_meta=support_meta,
                query_meta=query_meta,
                false_meta=None,
                task_emb=task_rel_emb,
            )
            scores.detach()
            scores = scores.data

            scores = scores.cpu().numpy()
            if save_and_return_scores is True:
                # Then we create a pandas dataframe here to analyze the scores.
                df = pd.DataFrame(
                    np.concatenate([query.cpu().numpy(), scores.reshape(-1, 1)], axis=1)
                )
                df.columns = ["head", "tail", "score"]
                df["is_real"] = False
                df.loc[0, "is_real"] = True
                df["relation"] = query_
                ret[query_]["queries"].append(df)
            sort = list(np.argsort(scores, kind="stable"))[::-1]
            rank = sort.index(0) + 1
            if rank <= 10:
                hits10.append(1.0)
                hits10_.append(1.0)
            else:
                hits10.append(0.0)
                hits10_.append(0.0)
            if rank <= 5:
                hits5.append(1.0)
                hits5_.append(1.0)
            else:
                hits5.append(0.0)
                hits5_.append(0.0)
            if rank <= 1:
                hits1.append(1.0)
                hits1_.append(1.0)
            else:
                hits1.append(0.0)
                hits1_.append(0.0)
            mrr.append(1.0 / rank)
            mrr_.append(1.0 / rank)
        
        logging.critical(
            "{} Hits10:{:.3f}, Hits5:{:.3f}, Hits1:{:.3f}, MRR:{:.3f}".format(
                query_,
                np.mean(hits10_),
                np.mean(hits5_),
                np.mean(hits1_),
                np.mean(mrr_),
            )
        )
        logging.info(
            "Number of candidates: {}, number of test examples {}".format(
                len(candidates), len(hits10_)
            )
        )
    logging.critical("HITS10: {:.3f}".format(np.mean(hits10)))
    logging.critical("HITS5: {:.3f}".format(np.mean(hits5)))
    logging.critical("HITS1: {:.3f}".format(np.mean(hits1)))
    logging.critical("MRR: {:.3f}".format(np.mean(mrr)))
    

    if save_and_return_scores is True:
        return np.mean(hits10), np.mean(hits5), np.mean(hits1), np.mean(mrr), ret
    return np.mean(hits10), np.mean(hits5), np.mean(hits1), np.mean(mrr)



def zeroshot_eval(model):
    pass
