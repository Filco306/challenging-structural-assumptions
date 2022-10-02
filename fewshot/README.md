# Few-shot



## Experiments

For running the experiments, we use [the code](https://github.com/Panda0406/Zero-shot-knowledge-graph-relational-learning) for the paper [Generative Adversarial Zero-Shot Relational Learning for Knowledge Graphs](https://arxiv.org/pdf/2001.02332.pdf)

To run experiments in the few-shot scenario, do the following: 

1. Download the code from [the original authors' repository](https://github.com/JiaweiSheng/FAAN). 
2. Follow their instructions and prepare the data folder, and make sure you can run their original code. 
3. Follow the instructions below to generate your own textual embeddings (or create your own using the data available).
4. One more modification is needed before running: at line 102 in `trainer.py`, add the name of your embedding (e.g., `uSIF`). This assumes you have generated `entity2vec.EMBEDDINGNAME` and `relation2vec.EMBEDDINGNAME` and that they lie in their corresponding dataset folders. 
5. Train your models with:
    - For **NELL-One**, do: `python trainer.py --weight_decay 0.0 --prefix nell.5shot --embed_model=EMBEDDING_NAME --embed_dim=EMBED_DIM` (if you do the paper experiments, `EMBEDDING_NAME`=`uSIF` and `EMBED_DIM`=`100`).
    - For **Wiki-One**, do: `python trainer.py --dataset wiki --embed_model EMBEDDING_NAME --embed_dim EMBED_DIM --num_transformer_layers 4 --num_transformer_heads 8 --dropout_input 0.3 --dropout_layers 0.2 --lr 6e-5 --prefix wiki.5shot` (if you do the paper experiments, `EMBEDDING_NAME`=`uSIF` and `EMBED_DIM`=`100`)


**IMPORTANT:** Some of the files (e.g., those that are zipped inside Wiki_fewshot.zip) might be placed in a subfolder after unzipping. When I ran the few-shot experiments, all the files were in the same subfolder, looking like this:

```
data
├── NELL
|   ├── enttext2entids
|   ├── ent2ids
|   ├── ...
│   └── Other files...
└── Wiki
    ├── enttext2entids
    ├── ent2ids
    ├── enttext2id_ablation_no_description
    ├── enttext2id_ablation_no_instance
    ├── ...
    └── enttext2id_ablation_no_label
```

## Generating embeddings 

The vectors were generated using the `scripts/process_data/generate_sif_embeddings.py`. If you wish to regenerate these, you can run source scripts/process_data/generate_all_vectors.sh (make sure you stand in the main folder). Make sure you have the necessary data files available.