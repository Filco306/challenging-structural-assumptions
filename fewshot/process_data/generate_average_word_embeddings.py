import wordninja
import argparse
import numpy as np
import os
from str2bool import str2bool
import sys
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.trainer import seed_everything  # noqa: E402
from scripts.process_data.generate_sif_embeddings import (  # noqa: E402
    load_w2vec,
    load_ids,
    construct_ent_word,
    construct_rel_word,
)


def compute_avg_word_embs(args):
    seed_everything()
    # dataset = "NELL"
    dataset = args.dataset
    # use_extra_descs = ""
    use_extra_descs = args.use_extra_descs
    # entity_text_file = "enttext2entids"
    entity_text_file = args.entity_text_file
    ent2ids, ents_to_ret, rel2ids, rels_to_ret = load_ids(
        dataset, use_extra_descs=use_extra_descs, entity_text_file=entity_text_file
    )

    ids2ent2words = {}
    ids2rel2words = {}
    print(f"Constructing and preprocessing entity descriptions for {dataset}")
    for i, enttext in tqdm(ents_to_ret.items()):
        if (
            construct_ent_word(
                ents_to_ret[i], dataset=dataset, use_extra_descs=use_extra_descs
            )
            in ids2ent2words
        ):
            raise Exception(f"{i} already accounted for : {enttext}")
        ids2ent2words[i] = construct_ent_word(
            enttext, dataset=dataset, use_extra_descs=use_extra_descs
        )
    print(f"Constructing and preprocessing relation descriptions for {dataset}")
    for i, reltext in tqdm(rels_to_ret.items()):
        ids2rel2words[i] = construct_rel_word(
            reltext, dataset=dataset, use_extra_descs=use_extra_descs
        )
    # glove = load_w2vec("glove-wiki-gigaword-100")
    glove = load_w2vec(args.w2vec_path)
    dim_ = glove["hello"].shape[0]
    print("Performing average! ")
    entity_vecs = np.zeros((len(ent2ids), dim_))
    rel_vecs = np.zeros((len(rel2ids), dim_))
    unknown_vec = np.random.randn(dim_)
    for i, text in tqdm(ids2ent2words.items()):
        vecs = [
            glove[x].reshape(-1, 1)
            for x in wordninja.split(text)
            if glove.vocab.get(text) is not None
        ]
        if len(vecs) > 0:
            vec = np.mean(np.concatenate(vecs, axis=1), axis=1)
        else:
            vec = unknown_vec
        entity_vecs[i] = vec
    for i, text in tqdm(ids2rel2words.items()):
        vecs = [
            glove[x].reshape(-1, 1)
            for x in wordninja.split(text)
            if glove.vocab.get(text) is not None
        ]
        if len(vecs) > 0:
            vec = np.mean(np.concatenate(vecs, axis=1), axis=1)
        else:
            vec = unknown_vec
        rel_vecs[i] = vec
    print("saving!")
    np.savetxt(
        "data/{}/entity2vec.average_embs{}".format(args.dataset, args.use_extra_descs),
        entity_vecs,
    )
    np.savetxt(
        "data/{}/relation2vec.average_embs{}".format(
            args.dataset, args.use_extra_descs
        ),
        rel_vecs,
    )


parser = argparse.ArgumentParser()

parser.add_argument(
    "--w2vec_path",
    type=str,
    default="glove-wiki-gigaword-100",
    help="""Name of w2vec embeddings. Can be 'glove-wiki-gigaword-100', 'glove-wiki-gigaword-50'
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

if __name__ == "__main__":
    args = parser.parse_args()
    print("generating!")
    compute_avg_word_embs(args)
