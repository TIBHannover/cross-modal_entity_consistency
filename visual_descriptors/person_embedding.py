import logging
import numpy as np
import os
from PIL.Image import open as open_image
from scipy.cluster.hierarchy import fclusterdata
import sys

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CUR_DIR, "contributed"))
from face import Detection, Encoder


class FacialFeatureExtractor:
    def __init__(self, model_path):
        self._FaceDetector = Detection()
        self._Encoder = Encoder(model_path)

    def get_img_embedding(self, image_path):
        try:
            image = open_image(image_path).convert("RGB")
            image = np.array(image)
            faces = self._FaceDetector.find_faces(image)

            embeddings = []
            for face in faces:
                embeddings.append(self._Encoder.generate_embedding(face))

            return embeddings
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logging.error(f"Cannot create embedding for {image_path}! {e}")
            return []


def agglomerative_clustering(embeddings, cluster_threshold=0.35, metric="cosine"):
    # NOTE: evaluated optimal threshold on LFW for face recognition is cosine distance in range [-1, 1] = 0.35
    # perform agglomerative clustering
    clusters = fclusterdata(X=embeddings, t=cluster_threshold, criterion="distance", metric=metric)

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
