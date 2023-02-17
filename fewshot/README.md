# FewShot experiments with FAAN

__DISCLAIMER__: I want to acknowledge the work of the original authors of the [FAAN](https://github.com/JiaweiSheng/FAAN) architecture. Please make sure to cite [their paper, Adaptive Attentional Network for Few-Shot Knowledge Graph Completion](https://aclanthology.org/2020.emnlp-main.131/) if you use this code, since I am re-using their architecture. 

## Run experiments 

To run experiments, do

```
python3 trainer.py --dataset "NELL" --embed_dim 100 --train_few 5 --few 5 --batch_size 128 --neg_num 1 --lr 0.00005 --margin 5.0 --dropout_input 0.3 --dropout_layers 0.2 --dropout_neighbors 0 --process_steps 2 --log_every 50 --eval_every 10000 --max_neighbor 50 --test --embed_model "uSIF" --prefix "FAAN_5Shot_uSIF_NELL" --loss "origin" --num_transformer_layers 3 --num_transformer_heads 4 --warm_up_step 10000 --max_batches 300000 --weight_decay 0 --grad_clip 5

python3 trainer.py --dataset "Wiki" --embed_dim 100 --train_few 5 --few 5 --batch_size 128 --neg_num 1 --lr 0.00006 --margin 5 --dropout_input 0.3 --dropout_layers 0.2 --dropout_neighbors 0 --process_steps 2 --log_every 50 --eval_every 400000 --max_neighbor 50 --test --embed_model uSIF --prefix "wiki.5shot.uSIF" --loss "origin" --num_transformer_layers 4 --num_transformer_heads 8 --warm_up_step 10000 --max_batches 300000 --weight_decay 0.0001 --grad_clip 5

```

## Generating embeddings
The vectors were generated using the `scripts/process_data/generate_sif_embeddings.py`. If you wish to regenerate these, you can run source `scripts/process_data/generate_all_vectors.sh` (make sure you stand in the main folder). Make sure you have the necessary data files available.
