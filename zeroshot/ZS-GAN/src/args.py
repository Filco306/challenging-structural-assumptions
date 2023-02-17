# -*- coding: utf-8 -*-
import argparse
import os
import json
import logging


def read_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="baseline_NELL.json", type=str)

    args = parser.parse_args()
    fp = os.path.join("config", args.config_file)
    assert os.path.exists(fp), f"File {fp} does not exist"

    with open(fp, "r") as f:
        args = json.load(f)

    args["save_path"] = "models/" + args["prefix"]

    logging.info("------HYPERPARAMETERS-------")
    for k, v in args.items():
        logging.info(k + ": " + str(v))
    logging.info("----------------------------")
    return args


if __name__ == "__main__":
    read_options()
