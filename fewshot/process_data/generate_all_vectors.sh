# !bin/bash

python3 scripts/process_data/create_wiki_all_descriptions.py
python3 scripts/process_data/generate_sif_embeddings.py --dataset="NELL"
python3 scripts/process_data/generate_sif_embeddings.py --dataset="Wiki"
python3 scripts/process_data/generate_sif_embeddings.py --dataset="Wiki" --w2vec_path="glove-wiki-gigaword-50" --use_extra_descs="_50d"
source scripts/process_data/generate_ablation_study_vectors.sh
python3 scripts/process_data/generate_transformer_embs.py --dataset="NELL"
python3 scripts/process_data/generate_transformer_embs.py --dataset="Wiki"