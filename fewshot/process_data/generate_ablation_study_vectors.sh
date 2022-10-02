# !bin/bash
python3 scripts/process_data/generate_sif_embeddings.py --dataset="NELL" --use_extra_descs="ablation_remove_type" --ablation_remove="type"
python3 scripts/process_data/generate_sif_embeddings.py --dataset="NELL" --use_extra_descs="ablation_remove_label" --ablation_remove="label"
python3 scripts/process_data/generate_sif_embeddings.py --dataset="Wiki" --use_extra_descs="ablation_no_description" --entity_text_file="enttext2id_ablation_no_description"
python3 scripts/process_data/generate_sif_embeddings.py --dataset="Wiki" --use_extra_descs="ablation_no_label" --entity_text_file="enttext2id_ablation_no_label"
python3 scripts/process_data/generate_sif_embeddings.py --dataset="Wiki" --use_extra_descs="ablation_no_instance" --entity_text_file="enttext2id_ablation_no_instance"
