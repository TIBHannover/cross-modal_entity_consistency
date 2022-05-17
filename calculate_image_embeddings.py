import argparse
import h5py
import json
import logging
import math
import multiprocessing
import numpy as np
import os
import sys
import yaml

# own imports
import utils
import metrics
from word_embedder import WordEmbedder


def parse_args():
    parser = argparse.ArgumentParser(description="inference script")
    parser.add_argument("-vv", "--debug", action="store_true", help="debug output")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--fasttext", type=str, required=True, help="Path to fasttext folder")
    parser.add_argument("-o", "--output", type=str, required=False, help="path to output folder")
    parser.add_argument("-t", "--threads", type=int, default=8, required=False, help="number of threads")
    args = parser.parse_args()
    return args


def calculate_results(x):
    doc, test_doc_ids, scene_word_embeddings, config = x

    if doc["id"] not in test_doc_ids:
        return None

    # read features
    features = config["features"]

    # structure is id/logits
    news_word_embeddings = h5py.File(features["word_embeddings"], "r")
    scene_logits = h5py.File(features["scene_logits"], "r")

    # get word embedding of the document
    if doc["id"] not in news_word_embeddings:
        logging.error(f"Cannot find word embeddings for news article: {doc['id']}")

    doc_emb = np.asarray(news_word_embeddings[doc["id"]], dtype=np.float32)  # |W| x |F_w|
    if doc_emb.shape[0] == 0:
        logging.info(f"No valid word embedding for {doc['id']}")
        return None

    testkey = "test_" + str(os.path.basename(config["split"]).split("_")[0])

    # calculate the cosine similarity of the noun word embeddings to all scene words in Places365
    word_scene_similarity = metrics.cossim(doc_emb, scene_word_embeddings)

    # calculate similarity for each set of entities
    document_results = {}
    for entityset in doc[testkey]:  # untampered, random, ...
        set_doc_id = doc[testkey][entityset][0]  # NOTE: just document id is stored

        # check if features for document image exist
        if set_doc_id not in scene_logits:
            logging.error(f"Cannot find scene logits for news article: {set_doc_id}")

        # get visual scene probabilities
        doc_scene_logits = np.expand_dims(np.asarray(scene_logits[set_doc_id], dtype=np.float32), axis=0)

        # calculate weighted context similarity for each word in the text
        context_similarities = np.sum(word_scene_similarity * doc_scene_logits, axis=-1)
        if math.isnan(np.max(context_similarities)):
            logging.info(f"NaN result for {doc['id']} set {entityset}")
            return None

        if config["operator"] == "max":
            document_results[entityset] = np.max(context_similarities)
        else:  # quantile with x percent
            document_results[entityset] = np.quantile(context_similarities, float(config["operator"][1:]) / 100)

    return document_results


def read_scene_labels(label_file):
    scene_labels = list()
    with open(label_file, "r") as class_file:
        for line in class_file:
            cls_name = line.strip().split(" ")[0][3:]
            cls_name = cls_name.split("/")[0]
            scene_labels.append(cls_name)

    return scene_labels


def get_scene_word_embeddings(classes, fasttext_bin_folder, language="en"):
    we = WordEmbedder(fasttext_bin_folder, token_types=None, language=language)
    scene_word_embeddings = []

    for cls in classes:
        cls_emb = we.generate_embeddings(cls)
        scene_word_embeddings.append(cls_emb)

    return np.concatenate(scene_word_embeddings, axis=0)


def main():
    # load arguments
    args = parse_args()

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # define logging level and format
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)

    # load dataset
    dataset = utils.read_jsonl(config["dataset"], dict_key="id")

    # create testset based on split or specified config params
    if config["split"]:
        # load splits (if specified)
        test_docs = dataset
        test_doc_ids = utils.read_split(config["split"])
    else:
        # generate testset based on given entity type and scenes type (indoor, outdoor) to consider
        test_docs = utils.generate_testset(dataset, entity_type=config["entity_type"], scenes=config["scenes"])
        test_doc_ids = set(test_docs.keys())

    logging.info(f"Number of test documents: {len(test_doc_ids)}")

    # create word embeddings for scene labels
    if os.path.basename(config["scene_labels"]) == "places365_en.txt":
        language = "en"
    else:  # places365_de.txt
        language = "de"

    logging.info("Generate word embedding for scene labels ...")
    scene_labels = read_scene_labels(config["scene_labels"])
    scene_word_embeddings = get_scene_word_embeddings(
        scene_labels, fasttext_bin_folder=args.fasttext, language=language
    )

    # generate results for each document
    document_similarities = []
    testset_similarities = {}
    with multiprocessing.Pool(args.threads) as p:
        pool_args = [(doc, test_doc_ids, scene_word_embeddings, config) for doc in dataset.values()]

        cnt_docs = 0
        for document_result in p.imap(calculate_results, pool_args):
            if document_result is None:
                continue

            document_similarities.append(document_result)

            cnt_docs += 1
            if cnt_docs % 100 == 0:
                logging.info(f"{cnt_docs} / {len(test_doc_ids)} documents processed ...")

            for key, val in document_result.items():
                if key not in testset_similarities:
                    testset_similarities[key] = []

                testset_similarities[key].append(val)

    results = metrics.calculate_metrics(testset_similarities)
    metrics.print_results(results)

    # save document similarities
    if args.output:
        if not os.path.exists(args.output):
            os.makedirs(args.output)

        testname = os.path.splitext(os.path.basename(args.config))[1]
        with open(os.path.join(args.output, testname + ".jsonl"), "w") as jsonlfile:
            for entry in document_similarities:
                jsonlfile.write(json.dumps(entry) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
