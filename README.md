# Challenging the Assumption of Structure-based embeddings in Few- and Zero-shot Knowledge Graph Completion

This is the code accompanying the paper `Challenging the Assumption of Structure-based embeddings in Few- and Zero-shot Knowledge Graph Completion`, published at LREC 2022. 

## Data 

The data resource is avaiable [here, through Google Drive](https://drive.google.com/drive/folders/1VM3KzQNL8qH-H4EjY_S6_63mX1j0mRwT?usp=sharing). 


## Experiments

See each respective folder. 

## Acknowledgements

For my experiments, I would as like to acknowledge the original authors of the papers: 
- [Adaptive Attentional Network for Few-Shot
Knowledge Graph Completion (FAAN)](https://www.aclweb.org/anthology/2020.emnlp-main.131.pdf) - [code here](https://github.com/JiaweiSheng/FAAN). 
- [Generative Adversarial Zero-Shot Relational Learning
for Knowledge Graphs (ZS-GAN)](https://arxiv.org/pdf/2001.02332.pdf) - [code here](https://github.com/Panda0406/Zero-shot-knowledge-graph-relational-learning).
- [OntoZSL: Ontology-enhanced Zero-shot Learning
 (OntoZSL)](https://dl.acm.org/doi/10.1145/3442381.3450042) - [code here](https://github.com/genggengcss/OntoZSL).

In my experiments, I re-used their code (instructions how to use these can be located in the READMEs of the respective folders). 

Also, I would like to acknowledge:
- The work of many contributors to Wikidata :) 
- The creators of NELL!

## Cite this work

If you use the additional textual data I collected for these tasks, don't forget to cite: 

```
@InProceedings{cornell-EtAl:2022:LREC,
  author    = {Cornell, Filip  and  zhang, Chenda  and  Karlgren, Jussi  and  Girdzijauskas, Sarunas},
  title     = {Challenging the Assumption of Structure-based embeddings in Few- and Zero-shot Knowledge Graph Completion},
  booktitle      = {Proceedings of the Language Resources and Evaluation Conference},
  month          = {June},
  year           = {2022},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages     = {6300--6309},
  abstract  = {In this paper, we report experiments on Few- and Zero-shot Knowledge Graph completion, where the objective is to add missing relational links between entities into an existing Knowledge Graph with few or no previous examples of the relation in question. While previous work has used pre-trained embeddings based on the structure of the graph as input for a neural network, nobody has, to the best of our knowledge, addressed the task by only using textual descriptive data associated with the entities and relations, much since current standard benchmark data sets lack such information. We therefore enrich the benchmark data sets for these tasks by collecting textual description data to provide a new resource for future research to bridge the gap between structural and textual Knowledge Graph completion. Our results show that we can improve the results for Knowledge Graph completion for both Few- and Zero-shot scenarios with up to a two-fold increase of all metrics in the Zero-shot setting. From a more general perspective, our experiments demonstrate the value of using textual resources to enrich more formal representations of human knowledge and in the utility of transfer learning from textual data and text collections to enrich and maintain knowledge resources.},
  url       = {https://aclanthology.org/2022.lrec-1.677}
}
```
 
Also, 
- When using the `NELL-ZS` and `Wiki-ZS`, don't forget to cite [Generative Adversarial Zero-Shot Relational Learning
for Knowledge Graphs](https://arxiv.org/pdf/2001.02332.pdf), or if you run any code from the submodule `Zero-shot-knowledge-graph-relational-learning`.
- When using the `NELL-One` and `Wiki-One` datasets, don't forget to cite [One-Shot Relational Learning for Knowledge Graphs](https://arxiv.org/pdf/1808.09040.pdf).
