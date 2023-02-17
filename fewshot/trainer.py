from collections import defaultdict
import argparse
from torch import optim
import torch
import numpy as np
import logging
from collections import deque
from data_loader import train_generate
from matcher import Matcher  # noqa: F401
from tensorboardX import SummaryWriter
import os
from dotenv import load_dotenv
from utils import load_embed, load_symbol2id, get_meta, build_connection
from evaluate import evaluate
import json
import random
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from args import read_options

class Trainer(object):
    
    def __init__(self, arg: dict):
        super(Trainer, self).__init__()
        if isinstance(arg, dict) is False:
            arg = arg.__dict__
        for k, v in arg.items():
            setattr(self, k, v)
        assert load_dotenv()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.meta = not self.no_meta
        self.use_relation_embeddings = arg.get("use_relation_embeddings", False)
        self.sample_in_eval = arg.get("sample_in_eval", 1.0)
        # pre-train
        if self.random_embed:
            use_pretrain = False
        else:
            use_pretrain = True

        logging.info("LOADING SYMBOL ID AND SYMBOL EMBEDDING")
        if self.test or self.random_embed:
            # gen symbol2id, without embedding
            self.symbol2id, self.symbol2vec = load_symbol2id(dataset=self.dataset)
            use_pretrain = False
        else:
            self.symbol2id, self.symbol2vec = load_embed(
                dataset=self.dataset, embed_model=self.embed_model
            )
        self.use_pretrain = use_pretrain

        self.num_symbols = len(self.symbol2id.keys()) - 1  # one for 'PAD'
        self.pad_id = self.num_symbols
        model_kwargs = {
            "embed_dim": self.embed_dim,
            "num_symbols": self.num_symbols,
            "use_pretrain": self.use_pretrain,
            "embed": self.symbol2vec,
            "dropout_layers": self.dropout_layers,
            "dropout_input": self.dropout_input,
            "dropout_neighbors": self.dropout_neighbors,
            "finetune": self.fine_tune,
            "num_transformer_layers": self.num_transformer_layers,
            "num_transformer_heads": self.num_transformer_heads,
            "device": self.device,
        }
        self.Matcher = eval(arg.get("model", "Matcher"))(**model_kwargs)

        self.Matcher.to(self.device)
        self.batch_nums = 0
        if self.test:
            self.writer = None
        else:
            self.writer = SummaryWriter("logs/" + self.prefix)

        self.parameters = filter(lambda p: p.requires_grad, self.Matcher.parameters())

        self.optim = optim.Adam(
            self.parameters, lr=self.lr, weight_decay=self.weight_decay
        )
        self.ent2id = json.load(open("data/" + self.dataset + "/ent2ids"))
        self.num_ents = len(self.ent2id.keys())

        logging.info("BUILDING CONNECTION MATRIX")
        self.connections, self.e1_rele2, self.e1_degrees = build_connection(
            ent2id=self.ent2id,
            symbol2id=self.symbol2id,
            dataset=self.dataset,
            num_ents=self.num_ents,
            pad_id=self.pad_id,
            max_=self.max_neighbor,
        )

        logging.info("LOADING CANDIDATES ENTITIES")
        self.rel2candidates = json.load(
            open("data/" + self.dataset + "/rel2candidates.json")
        )

        # load answer dict
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open("data/" + self.dataset + "/e1rel_e2.json"))

    def save(self, path=None):
        if not path:
            path = self.save_path
        torch.save(self.Matcher.state_dict(), path)

    def load(self, path=None):
        if path:
            self.Matcher.load_state_dict(torch.load(path))
        elif os.path.exists(self.save_path):
            self.Matcher.load_state_dict(torch.load(self.save_path))

    def train(self):
        logging.info("START TRAINING...")
        best_mrr = 0.0
        best_batches = 0

        losses = deque([], self.log_every)
        margins = deque([], self.log_every)
        for data in train_generate(
            self.dataset,
            self.batch_size,
            self.train_few,
            self.symbol2id,
            self.ent2id,
            self.e1rel_e2,
            self.use_relation_embeddings,
            self.embed_model,
        ):
            (
                support,
                query,
                false,
                support_left,
                support_right,
                query_left,
                query_right,
                false_left,
                false_right,
                task_rel_emb,
            ) = data

            self.batch_nums += 1
            support_meta = get_meta(
                left=support_left,
                right=support_right,
                connections=self.connections,
                e1_degrees=self.e1_degrees,
                device=self.device,
            )
            query_meta = get_meta(
                query_left,
                query_right,
                connections=self.connections,
                e1_degrees=self.e1_degrees,
                device=self.device,
            )
            false_meta = get_meta(
                false_left,
                false_right,
                connections=self.connections,
                e1_degrees=self.e1_degrees,
                device=self.device,
            )

            support = Variable(torch.LongTensor(support)).to(self.device)
            query = Variable(torch.LongTensor(query)).to(self.device)
            false = Variable(torch.LongTensor(false)).to(self.device)
            self.Matcher.train()
            if self.no_meta:
                positive_score, negative_score = self.Matcher(
                    support, query, false, isEval=False
                )
            else:
                positive_score, negative_score = self.Matcher(
                    support,
                    query,
                    false,
                    isEval=False,
                    support_meta=support_meta,
                    query_meta=query_meta,
                    false_meta=false_meta,
                    task_emb=task_rel_emb,
                )
            margin_ = positive_score - negative_score
            loss = F.relu(self.margin - margin_).mean()
            margins.append(margin_.mean().item())
            lr = adjust_learning_rate(
                optimizer=self.optim,
                epoch=self.batch_nums,
                lr=self.lr,
                warm_up_step=self.warm_up_step,
                max_update_step=self.max_batches,
            )
            losses.append(loss.item())

            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(self.parameters, self.grad_clip)
            self.optim.step()
            if self.batch_nums % self.log_every == 0:
                lr = self.optim.param_groups[0]["lr"]
                logging.info(
                    "Batch: {:d}, Avg_batch_loss: {:.6f}, lr: {:.6f}, ".format(
                        self.batch_nums, np.mean(losses), lr
                    )
                )
                self.writer.add_scalar(
                    "Avg_batch_loss_every_log", np.mean(losses), self.batch_nums
                )

            if self.batch_nums % self.eval_every == 0:
                logging.info("Batch_nums is %d" % self.batch_nums)
                hits10, hits5, hits1, mrr = evaluate(
                    trainer=self, meta=self.meta, mode="dev"
                )
                self.writer.add_scalar("HITS10", hits10, self.batch_nums)
                self.writer.add_scalar("HITS5", hits5, self.batch_nums)
                self.writer.add_scalar("HITS1", hits1, self.batch_nums)
                self.writer.add_scalar("MRR", mrr, self.batch_nums)
                self.save()

                if mrr > best_mrr:
                    self.save(self.save_path + "_best")
                    best_mrr = mrr
                    best_batches = self.batch_nums
                logging.info(
                    "Best_mrr is {:.6f}, when batch num is {:d}".format(
                        best_mrr, best_batches
                    )
                )

            if self.batch_nums == self.max_batches:
                self.save()
                break

            if self.batch_nums - best_batches > self.eval_every * 10:
                logging.info("Early stop!")
                self.save()
                break

    def test_(self, path=None):
        self.load(path)
        logging.info("Pre-trained model loaded for test")
        evaluate(self, mode="test", meta=self.meta)

    def eval_(self, path=None):
        self.load(path)
        logging.info("Pre-trained model loaded for dev")
        evaluate(self, mode="dev", meta=self.meta)


def seed_everything(seed=2040):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def adjust_learning_rate(
    optimizer,
    epoch,
    lr,
    warm_up_step,
    max_update_step,
    end_learning_rate=0.0,
    power=1.0,
):
    epoch += 1
    if warm_up_step > 0 and epoch <= warm_up_step:
        warm_up_factor = epoch / float(warm_up_step)
        lr = warm_up_factor * lr
    elif epoch >= max_update_step:
        lr = end_learning_rate
    else:
        lr_range = lr - end_learning_rate
        pct_remaining = 1 - (epoch - warm_up_step) / (max_update_step - warm_up_step)
        lr = lr_range * (pct_remaining ** power) + end_learning_rate

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr



def main_run(conf: dict):
    args = conf
    if not os.path.exists("./logs_"):
        os.mkdir("./logs_")
    if not os.path.exists("./logs"):
        os.mkdir("./logs")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler("./logs_/log-{}.txt".format(args["prefix"]))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    seed_everything(args["seed"])

    logging.info("*" * 100)
    logging.info("*** hyper-parameters ***")
    for k, v in args.items():
        logging.info(k + ": " + str(v))
    logging.info("*" * 100)

    trainer = Trainer(args)

    if args["test"]:
        trainer.test_()
        trainer.eval_()
    else:
        trainer.train()
        print("last checkpoint!")

        trainer.test_()
        trainer.eval_()
        print("best checkpoint!")
        trainer.eval_(args["save_path"] + "_best")
        trainer.test_(args["save_path"] + "_best")



def get_trainer(conf_file: str):
    assert conf_file[-5:] == ".json", "{} is not a json config file".format(conf_file)
    if conf_file.split("/")[0] != "config":
        conf_file = os.path.join("config", conf_file)
    conf = json.load(open(conf_file, "r"))
    _trainer = Trainer(conf)
    return _trainer


if __name__ == "__main__":
    conf = read_options().__dict__
    main_run(conf)
