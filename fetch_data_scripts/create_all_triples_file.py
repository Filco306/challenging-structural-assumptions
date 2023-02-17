import argparse
import json
import os

def flatten(li):
    return [item for sublist in li for item in sublist]

def main(args):
    bckkgpath = os.path.join(args.path_graph_path, "/path_graph.txt")
    bckkg = [x.split("\t") for x in open(bckkgpath, "r").readlines()]
    train_triples = json.load(open(args.path_graph_path + "/train_tasks.json", "r"))
    train_triples = flatten(train_triples.values())
    dev_triples = json.load(open(args.path_graph_path + "/dev_tasks.json", "r"))
    dev_triples = flatten(dev_triples.values())
    test_triples = json.load(open(args.path_graph_path + "/test_tasks.json", "r"))
    test_triples = flatten(test_triples.values())
    all_triples = bckkg + train_triples + dev_triples + test_triples
    all_triples = list(set([tuple(x) for x in all_triples]))
    with open(args.outfile, "w+") as f:
        for triple in all_triples:
            f.write("\t".join(triple) + "\n")
    print("Wrote {} triples to {}".format(len(all_triples), args.outfile))


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default="../data/Wiki")
    parser.add_argument("--outfile", type=str, default="./alltriples.txt")
    args = parser.parse_args()
    main(args)
