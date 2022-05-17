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
import metrics
import utils
from visual_descriptors.person_embedding import agglomerative_clustering
from word_embedder import WordEmbedder


def parse_args():
    parser = argparse.ArgumentParser(description="Test script for cross-modal entity similarity")
    parser.add_argument("-vv", "--debug", action="store_true", help="Debug output")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("-o", "--output", type=str, required=False, help="Path to output folder")
    parser.add_argument(
        "--fasttext", type=str, required=False, default="resources/fasttext", help="Path to fasttext folder"
    )
    parser.add_argument("-t", "--threads", type=int, default=8, required=False, help="Number of threads")
    args = parser.parse_args()
    return args


def get_entity_features(entities, features, sources, num_images, clustering, person_verification=False):
    entity_features = {}

    for entity in entities:
        if entity not in features:
            continue

        feature_vectors = []
        for engine in features[entity]:  # bing, google, wikidata
            if engine not in sources:
                continue

            engine_images = 0
            for entry in features[entity][engine]:
                fvec = np.asarray(features[entity][engine][entry])

                if person_verification:
                    # NOTE 1: for person verification an image could contain no faces, and thus no valid embedding
                    # person verification does not have to fulfill condition in NOTE 2
                    engine_images += 1

                if len(fvec.shape) < 2:  # NOTE 2: checks if image contains valid embeddings
                    continue

                if not person_verification:
                    # NOTE 3: the image (global features) itself is invalid if the condition in NOTE 2 is fullfilled
                    # each verification task (except person) need to fullfill the condition in NOTE 2
                    engine_images += 1

                if engine_images > num_images:
                    break

                feature_vectors.extend(fvec)

        if len(feature_vectors) < 1:
            logging.warning(f"No feature vector for entity: {entity}")
            return None

        if clustering:
            if len(feature_vectors) < 2:  # no clustering necessary
                logging.debug(f"No clustering for {entity} performed since embedding size is {len(feature_vectors)}...")
                entity_features[entity] = np.asarray(feature_vectors, dtype=np.float32)
            else:
                feature_vectors = agglomerative_clustering(feature_vectors)
                entity_features[entity] = np.asarray(feature_vectors, dtype=np.float32)
        else:
            entity_features[entity] = np.stack(feature_vectors, axis=0)

    return entity_features


def calculate_entity_cms(x):
    doc, test_doc_ids, config = x

    testkey = "test_" + config["entity_type"]

    if len(doc[testkey]["untampered"]) < 1:  # no untamperd entities for the given type
        return None

    if doc["id"] not in test_doc_ids:  # check if in intended split (if specified)
        return None

    # structure is id/feature(s)
    news_features = h5py.File(config["features"]["news_features"], "r")

    # structure is entity_wd_id/search_engine/feature(s)
    reference_features = h5py.File(config["features"]["entity_features"], "r")

    # get entity type
    if config["entity_type"] == "persons":  # Important for NOTE 1-3 in get_entity_features()
        person_verification = True
    else:
        person_verification = False

    # check if features for document image exist
    if doc["id"] not in news_features or len(news_features[doc["id"]].shape) < 2:
        logging.debug(f"Cannot find image features for news article: {doc['id']}")
        return None

    news_feature = np.asarray(news_features[doc["id"]], dtype=np.float32)

    # calculate similarity for each set of entities
    document_similarities = {}
    entity_similarities = {}
    for entityset in doc[testkey]:  # untampered, random, ...
        entity_features = get_entity_features(
            entities=doc[testkey][entityset],
            features=reference_features,
            sources=config["ref_images"]["sources"],
            num_images=config["ref_images"]["num_images"],
            clustering=config["clustering"],
            person_verification=person_verification,
        )

        if entity_features is None:
            return None

        entities_sims = []
        for entity, entity_feature in entity_features.items():
            entity_sims = metrics.cossim(news_feature, entity_feature)
            if config["operator"] == "max":
                entity_sim = np.max(entity_sims)
            else:  # quantile with x percent
                entity_sim = np.quantile(entity_sims, float(config["operator"][1:]) / 100)

            entities_sims.append(entity_sim)

            if entityset not in entity_similarities:
                entity_similarities[entityset] = {}

            entity_similarities[entityset][entity] = entity_sim

        if entityset not in document_similarities:
            document_similarities[entityset] = []
        document_similarities[entityset].append(np.max(entities_sims))

    return {"document_id": doc["id"], "similarities": document_similarities, "entity_similarities": entity_similarities}


def calculate_context_cms(x):
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
        logging.debug(f"No valid word embedding for {doc['id']}")
        return None

    testkey = "test_" + config["entity_type"]

    # calculate the cosine similarity of the noun word embeddings to all scene words in Places365
    word_scene_similarity = metrics.cossim(doc_emb, scene_word_embeddings)

    # calculate similarity for each set of entities
    document_similarities = {}
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
            logging.debug(f"NaN result for {doc['id']} set {entityset}")
            return None

        if config["operator"] == "max":
            document_similarities[entityset] = np.max(context_similarities)
        else:  # quantile with x percent
            document_similarities[entityset] = np.quantile(context_similarities, float(config["operator"][1:]) / 100)

    return {"document_id": doc["id"], "similarities": document_similarities}


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


def calculate_split_results(similarities, split_docs):
    testset_similarities = {}
    for doc in similarities:

        if doc["document_id"] not in split_docs:
            continue

        for key, val in doc["similarities"].items():
            if key not in testset_similarities:
                testset_similarities[key] = []

            testset_similarities[key].append(val)

    return metrics.calculate_metrics(testset_similarities)


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

    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )

    # load dataset
    dataset = utils.read_jsonl(config["dataset"], dict_key="id")

    # generate testset based on given entity type and scenes type (indoor, outdoor) to consider
    test_docs = utils.generate_testset(dataset, entity_type=config["entity_type"], scenes=config["scenes"])
    test_doc_ids = set(test_docs.keys())

    logging.info(f"Number of test documents: {len(test_doc_ids)}")

    # create word embeddings for scene labels
    if config["entity_type"] == "context":
        if os.path.basename(config["scene_labels"]) == "places365_en.txt":
            language = "en"
        else:  # places365_de.txt
            language = "de"

        logging.info("Generate word embedding for scene labels ...")
        scene_labels = read_scene_labels(config["scene_labels"])
        scene_word_embeddings = get_scene_word_embeddings(
            scene_labels, fasttext_bin_folder=args.fasttext, language=language
        )

        pool_args = [(doc, test_doc_ids, scene_word_embeddings, config) for doc in dataset.values()]
        cms_fn = calculate_context_cms
    else:
        pool_args = [(doc, test_doc_ids, config) for doc in test_docs.values()]
        cms_fn = calculate_entity_cms

    # generate similarities for each document
    document_similarities = []
    cnt_docs = 0

    with multiprocessing.Pool(args.threads) as p:
        for document in p.imap(cms_fn, pool_args):
            if document is None:
                continue

            document_similarities.append(document)

            cnt_docs += 1
            if cnt_docs % 100 == 0:
                logging.info(f"{cnt_docs} / {len(test_doc_ids)} documents processed ...")

    # calculate results for metrics based on specified split (percentage) of the dataset
    document_similarities_sorted = sorted(
        document_similarities, key=lambda k: k["similarities"]["untampered"], reverse=True
    )

    len_k = int(0.5 + len(document_similarities_sorted) * (config["percentage"] / 100))
    docs_at_k = document_similarities_sorted[:len_k]
    split_docs = set([x["document_id"] for x in docs_at_k])

    results = calculate_split_results(document_similarities_sorted, split_docs)
    metrics.print_results(results)

    # optional: save document similarities, splits, and results
    if args.output:
        if not os.path.exists(args.output):
            os.makedirs(args.output)

        testname = os.path.splitext(os.path.basename(args.config))[0]
        with open(os.path.join(args.output, testname + "_similarities.jsonl"), "w") as jsonlfile:
            for entry in document_similarities:
                jsonlfile.write(json.dumps(entry) + "\n")

        with open(os.path.join(args.output, testname + "_metrics.jsonl"), "w") as jsonlfile:
            for testset, result in results.items():
                jsonlfile.write(json.dumps({"testset": testset, "result": result}) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
