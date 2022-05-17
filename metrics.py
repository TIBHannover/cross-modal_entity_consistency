import logging
import numpy as np
import scipy.spatial as sp
from sklearn import metrics


def cossim(f1, f2):
    cos_dist = sp.distance.cdist(f1, f2, "cosine")  # 0 equal, 2 different
    norm_cos_dist = cos_dist / 2.0  # 0 equal, 1 different
    return 1 - norm_cos_dist  # 1 equal, 0 different


def calculate_metrics(testset_similarities):
    true_labels = np.ones(len(testset_similarities["untampered"]), dtype=int)
    true_similarities = np.asarray(testset_similarities["untampered"], dtype=np.float32)

    results = {}
    for key in testset_similarities:
        if key == "untampered":
            continue

        false_labels = np.zeros(len(testset_similarities[key]), dtype=int)
        false_similarities = np.asarray(testset_similarities[key], dtype=np.float32)

        labels = np.concatenate((true_labels, false_labels))
        similarities = np.concatenate((true_similarities, false_similarities))

        fpr, tpr, _ = metrics.roc_curve(labels, similarities, pos_label=1)
        auc_untamp = metrics.auc(fpr, tpr)

        # Ranking metrics
        cnt_ranking_correct = 0
        for i in range(len(true_similarities)):
            if true_similarities[i] > false_similarities[i]:
                cnt_ranking_correct += 1

        results[key] = {
            "first_rank_percentage": cnt_ranking_correct / len(true_similarities),
            "auc": auc_untamp,
            "AP_at25R_untampered": ap_at_kperc_recall(labels, similarities, 0.25),
            "AP_at50R_untampered": ap_at_kperc_recall(labels, similarities, 0.5),
            "AP_at100R_untampered": ap_at_kperc_recall(labels, similarities, 1),
            "AP_at25R_tampered": ap_at_kperc_recall(labels, similarities, 0.25, gt_label=0),
            "AP_at50R_tampered": ap_at_kperc_recall(labels, similarities, 0.5, gt_label=0),
            "AP_at100R_tampered": ap_at_kperc_recall(labels, similarities, 1, gt_label=0),
        }

    return results


def ap_at_kperc_recall(labels, similarities, kperc, gt_label=1):
    _, labels_sorted = (list(t) for t in zip(*sorted(zip(similarities, labels))))

    if gt_label == 1:
        labels_sorted = labels_sorted[::-1]

    labels_sorted = np.asarray(labels_sorted, dtype=np.uint8)

    relevant_docs = int(0.5 + np.sum(np.equal(labels_sorted, gt_label)) * kperc)
    if relevant_docs < 1:
        logging.warning(f"Not enough relevant documents for kperc = {kperc}")
        return [0, 0]

    cnt_correct = 0
    cnt_seen = 0
    sum_p = 0
    for label in labels_sorted:
        cnt_seen += 1
        if label == gt_label:
            cnt_correct += 1
            sum_p += cnt_correct / cnt_seen

        if cnt_correct == relevant_docs:
            break

    return [sum_p / cnt_correct, relevant_docs]


def print_results(results):
    for testset in results:
        logging.info("################################")
        logging.info(testset)
        logging.info("################################")
        for key, val in results[testset].items():
            if "AP" in key:
                val = val[0] * 100
            logging.info(f"{key}: {val:.2f}")
