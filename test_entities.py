import argparse
import h5py
import logging
import multiprocessing
import numpy as np
import os
from scipy.cluster.hierarchy import fclusterdata
import sys
import yaml

# own imports
import utils
import metrics


def parse_args():
    parser = argparse.ArgumentParser(description="inference script")
    parser.add_argument("-vv", "--debug", action="store_true", help="debug output")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("-t", "--threads", type=int, default=8, required=False, help="number of threads")
    args = parser.parse_args()
    return args


def get_entity_features(entities, features, num_images, clustering, person_verification=False):
    entity_features = {}
    for entity in entities:
        if entity not in features:
            continue

        feature_vectors = []
        for engine in features[entity]:  # bing, google, wikidata
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

        if clustering:
            if len(feature_vectors) < 2:  # no clustering necessary
                logging.debug(f'No clustering for {entity} performed since embedding size is {len(feature_vectors)}...')
                entity_features[entity] = np.asarray(feature_vectors, dtype=np.float32)
            else:
                feature_vectors = agglomerative_clustering(feature_vectors)
                entity_features[entity] = np.asarray(feature_vectors, dtype=np.float32)
        else:
            entity_features[entity] = np.stack(feature_vectors, axis=0)

    return entity_features


def agglomerative_clustering(embeddings, cluster_threshold=0.35, metric='cosine'):
    # NOTE: evaluated optimal threshold on LFW for face recognition is cosine distance in range [-1, 1] = 0.35
    # perform agglomerative clustering
    clusters = fclusterdata(X=embeddings, t=cluster_threshold, criterion='distance', metric=metric)

    # get majority cluster(s)
    count = np.bincount(clusters)
    max_clusters = [i for i, j in enumerate(count) if j == max(count)]

    # get mean embedding for majority cluster(s) and return
    filtered_dict = {}
    for i, emb in enumerate(embeddings):
        if clusters[i] in max_clusters:
            if clusters[i] in filtered_dict:
                filtered_dict[clusters[i]].append(emb)
            else:
                filtered_dict[clusters[i]] = [emb]

    filtered_embeddings = []
    for filtered_list in filtered_dict.values():
        emb = np.mean(filtered_list, axis=0)
        filtered_embeddings.append(emb)

    return filtered_embeddings


def calculate_results(x):
    doc, test_doc_ids, config = x

    if doc["id"] not in test_doc_ids:
        return None

    # load features
    features = config["features"]

    # structure is id/feature(s)
    news_features = h5py.File(features["news_features"], 'r')

    # structure is entity_wd_id/search_engine/feature(s)
    untampered_features = h5py.File(features["untampered_reference_features"], 'r')
    tampered_features = h5py.File(features["tampered_reference_features"], 'r')

    # get entity type
    if "persons" in config["split"]:  # Important for NOTE 1-3 in get_entity_features()
        person_verification = True
    else:
        person_verification = False
    testkey = "test_" + str(os.path.basename(config["split"]).split("_")[0])

    # check if features for document image exist
    if doc["id"] not in news_features:
        logging.error(f"Cannot find image features for news article: {doc['id']}")
    news_feature = np.asarray(news_features[doc["id"]], dtype=np.float32)

    # calculate similarity for each set of entities
    document_results = {}
    for entityset in doc[testkey]:  # untampered, random, ...
        if entityset == "untampered":
            features = untampered_features
        else:
            features = tampered_features

        entity_features = get_entity_features(entities=doc[testkey][entityset],
                                              features=features,
                                              num_images=config["num_images"],
                                              clustering=config["clustering"],
                                              person_verification=person_verification)

        entities_sims = []
        for entity, entity_feature in entity_features.items():
            entity_sims = metrics.cossim(news_feature, entity_feature)
            if config["operator"] == "max":
                entity_sim = np.max(entity_sims)
            else:  # quantile with x percent
                entity_sim = np.quantile(entity_sims, float(config["operator"][1:]) / 100)

            entities_sims.append(entity_sim)

        if entityset not in document_results:
            document_results[entityset] = []
        document_results[entityset].append(np.max(entities_sims))

    return document_results


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

    # load splits
    test_doc_ids = utils.read_split(config["split"])
    logging.info(f"Number of test documents: {len(test_doc_ids)}")

    # load dataset
    dataset = utils.read_jsonl(config["dataset"], dict_key="id")

    # generate results for each document
    testset_similarities = {}
    with multiprocessing.Pool(args.threads) as p:
        pool_args = [(doc, test_doc_ids, config) for doc in dataset.values()]

        cnt_docs = 0
        for document_result in p.imap(calculate_results, pool_args):
            if document_result is None:
                continue

            cnt_docs += 1
            if cnt_docs % 100 == 0:
                logging.info(f"{cnt_docs} / {len(test_doc_ids)} documents processed ...")

            for key, val in document_result.items():
                if key not in testset_similarities:
                    testset_similarities[key] = []

                testset_similarities[key].append(val)

    results = metrics.calculate_metrics(testset_similarities)
    metrics.print_results(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
