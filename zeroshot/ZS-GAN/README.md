# Zero-shot relation relation with text

__DISCLAIMER__: I am not the original author of this code, and this is a refactored version. All credits for developing the model architecture go to the original authors of ZS-GAN. If you use this code, make sure to cite [Generative Adversarial Zero-Shot Relational Learning for Knowledge Graphs](https://arxiv.org/pdf/2001.02332.pdf).

## Experiments, ZS-GAN

For running the experiments, we use [the code](https://github.com/Panda0406/Zero-shot-knowledge-graph-relational-learning) for the paper [Generative Adversarial Zero-Shot Relational Learning for Knowledge Graphs](https://arxiv.org/pdf/2001.02332.pdf) (see folder `ZS-GAN`).

To run the experiments for ZS-GAN, do the following:


1. Reproduce the baseline by running `python trainer.py --pretrain_feature_extractor` for NELL-ZS and `python3 trainer.py --pretrain_feature_extractor --grad_clip 5.0 --noise_dim 15 --dataset Wiki --D_epoch 5 --test_sample 20 --seed 0 --pretrain_margin 10.0 --G_epoch 1 --weight_decay 0.0 --pretrain_few 30 --REG_W 0.001 --train_times 6000 --pretrain_batch_size 128 --embed_dim 50 --pretrain_subepoch 30 --save_path models/intial --REG_Wz 0.0001 --gan_batch_rela 8 --embed_model DistMult --pretrain_loss_every 200 --loss_every 200 --dropout 0.5 --w_dim 50 --aggregate max --D_batch_size 64 --pretrain_times 7000 --device 0 --prefix intial --max_neighbor 50 --lr_G 0.0001 --lr_E 0.0005 --lr_D 0.0001 --G_batch_size 64` for Wiki-ZS. 
3. Now, to run your experiments with textual embeddings instead, do: 
    - For `NELL-ZS`, do: `python trainer.py --pretrain_feature_extractor --embed_dim=EMBED_DIM --w_dim=2*EMBED_DIM --embed_model=EMBED_MODEL` where 
        - `EMBED_DIM`=`384`, `EMBED_MODEL`=`BERT` or `EMBED_DIM`=`100`, `EMBED_MODEL`=``

## Generate embeddings for datasets

To generate embeddings for datasets, you need to:

1. Place the `origin_data/DATASET_NAME` in the `Zero-shot-knowledge-graph-relational-learning` folder.
2. Make sure you have the following files in `origin_data/DATASET_NAME`:
   - `entity2id`
   - `relation2ids`
   - `relationdescriptions`
   - `enttext2ids`
3. Run `python3 scripts/process_data/generate_sif_embs_for_ZS_data.py`
4. Run `python3 scripts/process_data/generate_sif_embeddings.py --dataset=DATASET_NAME --entity_text_file="enttext2ids"`. The options for `DATASET_NAME` are either `Wiki` or `NELL`.


## Experiments, OntoZSL

For running the experiments, we use [the code](https://github.com/genggengcss/OntoZSL) for the paper [OntoZSL: Ontology-enhanced Zero-shot Learning
](https://dl.acm.org/doi/10.1145/3442381.3450042) (see folder `OntoZSL`).

To run the experiments for OntoZSL, do the following:

1. Follow their [OntoZSL's original instructions](https://github.com/genggengcss/OntoZSL), and make sure you can run their experiments. Some reminders here: 
    - You need to train the `OntoEncoder` for that; see the instructions provided by the original authors in `OntoZSL/code/OntoEncoder/README.md`. 
2. Place the embeddings you produced for ZS-GAN in the data, and then run: 
```python3
# Wiki
python gan_kgc.py --dataset Wiki --embed_model EMBED_MODEL --splitname ori --embed_dim EMBED_DIM --ep_dim EMBED_DIM*2 --fc1_dim 200 --pretrain_batch_size 128 --pretrain_subepoch 30 --pretrain_times 7000 --D_batch_size 64 --G_batch_size 64 --gan_batch_rela 8   --load_trained_embed --semantic_of_rel rela_matrix_onto_wiki.npz --device 0 --pretrain_feature_extractor
# NELL
python gan_kgc.py --dataset NELL --embed_model EMBED_MODEL --splitname ori --embed_dim EMBED_DIM --ep_dim EMBED_DIM*2 --fc1_dim 400 --pretrain_batch_size 64 --pretrain_subepoch 20 --pretrain_times 16000 --D_batch_size 256 --G_batch_size 256 --gan_batch_rela 2 --pretrain_feature_extractor --load_trained_embed --semantic_of_rel rela_matrix_onto_nell.npz --device 0
```
