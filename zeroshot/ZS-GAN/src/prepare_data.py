# -*- coding: utf-8 -*-
import re
import numpy as np
from collections import defaultdict
from args import read_options
import pickle
import json
from PIL import ImageFile
import logging
import io
import requests
from PIL import Image
from torchvision import transforms

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


ImageFile.LOAD_TRUNCATED_IMAGES = True

data_path = "./origin_data/"


def preprocess_sent(sent, max_length, word2id):
    sent = sent.strip().split()
    sent = sent[:max_length]
    sent2id = np.zeros(max_length, dtype=np.int32)

    for i in range(len(sent)):
        if sent[i] not in word2id:
            sent2id[i] = word2id["UNK"]
        else:
            sent2id[i] = word2id[sent[i]]

    return sent2id


def preprocess_img(image_url):
    response = requests.get(image_url)
    img_pil = Image.open(
        io.BytesIO(response.content)
    )  # it can be displayed in ipython notebook

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    #  output a Tensor of size [3, 224, 224]
    return preprocess(img_pil)


class Expand_query_Wiki(object):
    """
    Extract availabel infomation from internet
    Qid: the list of query ids
    """

    def __init__(self, Qid, client, args, word2id):
        self.Qid = Qid
        self.client = client
        self.entity = self.client.get(self.Qid, load=True)
        self.image_prop = self.client.get("P18")
        self.word2id = word2id
        self.args = args

        self.props = [str(k) for k in self.entity.keys()]
        self.count_image_file = 0

    def description(self):
        return str(self.entity.description)

    def image_resolution(self):
        image = self.entity[self.image_prop]
        return image.image_resolution

    def image_size(self):
        image = self.entity[self.image_prop]
        return image.image_size

    def image_url(self):
        image = self.entity[self.image_prop]
        return image.image_url

    def image_tensor(self):
        if str(self.image_prop) in self.props:
            image_url = self.image_url()
            try:
                image_Tensor = preprocess_img(image_url)
            except OSError as e:  # noqa: F841
                image_Tensor = "Failure!"
            return image_Tensor
        else:
            return "None"

    def description_ids(self):
        sent = self.description()
        if len(sent) != 0:
            return preprocess_sent(sent, self.args["max_sent"], self.word2id)
        else:
            return "None"


def relation_description_NELL(rela2id, dataset):
    f_description = open(
        "./origin_data/" + dataset + "/" + dataset + "_description"
    ).readlines()

    descriptions = dict()
    for line in f_description:
        s = line.strip().split("\t")
        relation = s[0]
        text_ = s[2]
        text_ = clean_str(text_)
        descriptions[relation] = text_

    rela2description = dict()

    for relation in rela2id.keys():
        relation_name = relation.split(":")[1]
        relation_name = relation_name.split("_")[0]
        rela2description[relation] = descriptions[relation_name]

    json.dump(rela2description, open("../rela2description.json", "w"))

    # noreap = set()
    # for k,v in rela2description.items():
    #    noreap.add(v)
    # logging.info(len(noreap))
    return rela2description


def generate_matrix_(name2id, name2text, max_length, word2id):
    name_num = len(name2id)
    matrix = np.zeros((name_num, max_length), dtype=np.int32)
    for name, idx in name2id.items():
        sent = name2text[name]
        sent2id = preprocess_sent(sent, max_length, word2id)
        matrix[idx] = sent2id

    return matrix


def generate_matrix(WordMatrix, corpus, corpus_tfidf, rela_list, w_dim=300):
    doc_Matrix = np.zeros(shape=(len(corpus), w_dim), dtype="float32")
    for num in range(len(corpus)):
        doc = corpus[num]
        doc_tfidf = corpus_tfidf[num]
        tmp_vec = np.zeros(shape=(w_dim,), dtype="float32")
        non_repeat = list()
        for i in range(len(doc)):
            # logging.info '1', WordMatrix[doc[i]]
            # logging.info doc_tfidf[i]
            # logging.info '2', WordMatrix[doc[i]] * doc_tfidf[i]
            if doc[i] not in non_repeat:
                non_repeat.append(doc[i])
                tmp_vec += WordMatrix[doc[i]] * doc_tfidf[i]
        logging.info(rela_list[num], len(non_repeat), len(doc))
        tmp_vec = tmp_vec / float(len(non_repeat))
        doc_Matrix[num] = tmp_vec
        # break

    return doc_Matrix


def get_vocabulary(rela2text):
    vocab = defaultdict(float)
    for rela, text in rela2text.items():
        text_ = text.split()
        for word in text_:
            vocab[word] += 1
    return vocab


def load_wordembedding_50(data_path, vocab, dataname="Wiki"):
    fvec = "./origin_data/vec.txt"
    word2id, word_vecs = dict(), dict()

    with open(fvec, "r") as f:
        W_size, w_dim = f.readline().strip().split()[:2]
        W_size, w_dim = int(W_size), int(w_dim)

        for i in range(W_size):
            temp = f.readline().strip().split()
            word = temp[0]
            if word in list(vocab.keys()):
                vec = np.zeros(w_dim, dtype="float32")
                for j in range(w_dim):
                    vec[j] = (float)(temp[j + 1])
                word_vecs[word] = vec

    W = np.zeros(shape=(len(word_vecs) + 2, w_dim), dtype="float32")
    word2id["BLANK"] = 0  # the padding vector
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word2id[word] = i
        i += 1

    word2id["UNK"] = len(word_vecs) + 1
    W[len(word_vecs) + 1] = np.random.uniform(-0.25, 0.25, w_dim)  # UNK
    pickle.dump(
        word2id, open(data_path + dataname + "/word2id_50_" + dataname + ".pkl", "wb")
    )
    np.savez(data_path + dataname + "/WordMatrix_50_" + dataname, W)
    logging.info(
        "Dataset %s ---- word2id size: %d, word matrix size: %s"
        % (dataname, len(word2id), str(W.shape))
    )

    return word2id


def load_wordembedding_300(data_path, vocab, dataname="Wiki"):
    fvec = "/home/pengda/GoogleNews-vectors-negative300.bin"
    word2id, word_vecs = dict(), dict()

    with open(fvec, "rb") as f:
        header = f.readline()
        W_size, w_dim = map(int, header.split())
        binary_len = np.dtype("float32").itemsize * w_dim
        for line in range(W_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == " ":
                    word = "".join(word)
                    break
                if ch != "\n":
                    word.append(ch)
            if word in list(vocab.keys()):
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype="float32")
            else:
                f.read(binary_len)
    W = np.zeros(shape=(len(word_vecs), w_dim), dtype="float32")
    i = 0
    for word in word_vecs:
        W[i] = word_vecs[word]
        word2id[word] = i
        i += 1
    # logging.info 'risks:', word_vecs['risks']
    # logging.info 'all', word_vecs['all']

    pickle.dump(
        word2id, open(data_path + dataname + "/word2id_300_" + dataname + ".pkl", "wb")
    )
    np.savez(data_path + dataname + "/WordMatrix_300_" + dataname, W)
    logging.info(
        "Dataset %s ---- word2id size: %d, word matrix size: %s"
        % (dataname, len(word2id), str(W.shape))
    )

    return word2id, W


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets.
    Every dataset is lower cased
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\.]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_OOV(rela2doc, word2id):
    rela_list = list()
    corpus = list()
    corpus_text = list()
    for rela, doc in rela2doc.items():
        rela_list.append(rela)
        doc = doc.split()
        cleaned_doc_id = list()
        cleaned_doc = list()
        for word in doc:
            if word in word2id.keys():  # word2id.has_key(word):
                cleaned_doc_id.append(word2id[word])
                cleaned_doc.append(word)
            # else:
            #    logging.info word,
        corpus.append(cleaned_doc_id)
        corpus_text.append(" ".join(cleaned_doc))
        # logging.info '\n'
    return rela_list, corpus_text


def calculate_tfidf_(rela_list, corpus, word2id):
    tfidf_vec = TfidfVectorizer(stop_words=stopwords.words("english"))
    # transformer=TfidfTransformer(stop_words=stopwords.words('english'))
    tfidf = tfidf_vec.fit_transform(corpus)
    word = tfidf_vec.get_feature_names()  # list, num of words
    weight = tfidf.toarray()  # (181, num of words)
    weight = weight.astype("float32")

    corpus_tfidf = list()
    corpus_new = list()
    for num in range(len(rela_list)):
        word2tfidf = zip(word, list(weight[num]))
        word2tfidf = dict(word2tfidf)
        assert len(word) == len(list(weight[num]))
        # doc_tfidf = list()
        # doc_ids = list()
        word_list = corpus[num].split()
        w2t = dict()
        for w in word_list:
            if w in word:
                w2t[word2id[w]] = word2tfidf[w]
        tmp = list(set(w2t.items()))
        w2t_sorted_ = sorted(tmp, key=lambda x: x[1], reverse=True)
        w2t_sorted = w2t_sorted_[:20]

        doc_tfidf = [item[1] for item in w2t_sorted]
        doc_ids = [item[0] for item in w2t_sorted]

        corpus_tfidf.append(doc_tfidf)
        corpus_new.append(doc_ids)

        logging.info("RELATION: ", rela_list[num])
        logging.info(w2t_sorted[:10], "\n")

    assert len(corpus_tfidf) == len(rela_list)

    return corpus_tfidf, word, corpus_new


def calculate_tfidf(rela_list, corpus, word2id):
    tfidf_vec = TfidfVectorizer(stop_words=stopwords.words("english"))
    tfidf = tfidf_vec.fit_transform(corpus)
    word = tfidf_vec.get_feature_names()  # list, num of words
    weight = tfidf.toarray()  # (181, num of words)
    weight = weight.astype("float32")

    corpus_tfidf = list()
    corpus_new = list()
    for num in range(len(rela_list)):
        word2tfidf = zip(word, list(weight[num]))
        word2tfidf = dict(word2tfidf)
        assert len(word) == len(list(weight[num]))
        doc_tfidf = list()
        doc_ids = list()
        word_list = corpus[num].split()
        for w in word_list:
            if w in word:
                doc_tfidf.append(word2tfidf[w])
                doc_ids.append(word2id[w])

        corpus_tfidf.append(doc_tfidf)
        corpus_new.append(doc_ids)

    assert len(corpus_tfidf) == len(rela_list)

    return corpus_tfidf, word, corpus_new


def NELL_text_embedding(args, dataname="NELL"):
    # read rela_document.txt
    rela2doc = dict()
    with open(data_path + dataname + "/rela_document.txt") as f_doc:
        lines = f_doc.readlines()
        for num in range(181):
            rela = lines[5 * num].strip().split("###")[0].strip()
            description = lines[5 * num + 1].strip().split("##")[1].strip()
            description = clean_str(description)
            e1 = lines[5 * num + 2].strip().split("###")[1].strip()
            e1 = clean_str(e1)
            e2 = lines[5 * num + 3].strip().split("###")[1].strip()
            e2 = clean_str(e2)
            rela2doc[rela] = (
                e1
                + " "
                + description
                + " "
                + description
                + " "
                + description
                + " "
                + description
                + " "
                + description
                + " "
                + e2
            )

    # Generate NELL description vocabulary
    vocab = get_vocabulary(rela2doc)
    logging.info("NELL description vocab size %d" % (len(vocab)))

    word2id, WordMatrix = load_wordembedding_300(data_path, vocab, dataname="NELL")
    # WordMatrix = np.load('./origin_data/NELL/WordMatrix_300_NELL.npz')['arr_0']
    # word2id = pickle.load(open('./origin_data/NELL/word2id_300_NELL.pkl'))

    rela_list, corpus_text = clean_OOV(rela2doc, word2id)

    reldes2ids = dict()
    for i, rela in enumerate(rela_list):
        reldes2ids[rela] = int(i)
    json.dump(reldes2ids, open(data_path + dataname + "/reldes2ids", "w"))

    corpus_tfidf, vocab_tfidf, corpus = calculate_tfidf(rela_list, corpus_text, word2id)

    rela_matrix_NELL = generate_matrix(WordMatrix, corpus, corpus_tfidf, rela_list)
    logging.info(rela_matrix_NELL)
    np.savez(data_path + dataname + "/rela_matrix", relaM=rela_matrix_NELL)
    logging.info("rela_matrix shape %s" % (str(rela_matrix_NELL.shape)))


if __name__ == "__main__":
    args = read_options()

    # read rela_document.txt
    rela2doc = dict()
    with open("./origin_data/NELL/rela_document.txt") as f_doc:
        lines = f_doc.readlines()
        for num in range(181):
            rela = lines[5 * num].strip().split("###")[0].strip()
            description = lines[5 * num + 1].strip().split("##")[1].strip()
            description = clean_str(description)
            e1 = lines[5 * num + 2].strip().split("###")[1].strip()
            e1 = clean_str(e1)
            e2 = lines[5 * num + 3].strip().split("###")[1].strip()
            e2 = clean_str(e2)
            rela2doc[rela] = (
                e1
                + " "
                + description
                + " "
                + description
                + " "
                + description
                + " "
                + description
                + " "
                + description
                + " "
                + e2
            )

    # Generate NELL description vocabulary

    vocab = get_vocabulary(rela2doc)
    logging.info("NELL description vocab size %d" % (len(vocab)))

    WordMatrix = np.load("./data/NELL/WordMatrix_300_NELL.npz")["arr_0"]
    word2id = pickle.load(open("./data/NELL/word2id_300_NELL.pkl", "rb"))

    rela_list, corpus_text = clean_OOV(rela2doc, word2id)

    reldes2ids = dict()
    for i, rela in enumerate(rela_list):
        reldes2ids[rela] = int(i)
    json.dump(reldes2ids, open("./data/NELL/reldes2ids", "w"))

    corpus_tfidf, vocab_tfidf, corpus = calculate_tfidf(rela_list, corpus_text, word2id)

    rela_matrix_NELL = generate_matrix(WordMatrix, corpus, corpus_tfidf, rela_list)
    logging.info(rela_matrix_NELL)
    np.savez("./data/NELL/rela_matrix", relaM=rela_matrix_NELL)
    logging.info("rela_matrix shape %s" % (str(rela_matrix_NELL.shape)))
