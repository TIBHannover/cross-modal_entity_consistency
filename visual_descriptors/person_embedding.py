import logging
import numpy as np
import os
from PIL.Image import open as open_image
import sys

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CUR_DIR, 'contributed'))
from face import Detection, Encoder


class FacialFeatureExtractor():

    def __init__(self, model_path):
        self._FaceDetector = Detection()
        self._Encoder = Encoder(model_path)

    def get_img_embedding(self, image_path):
        try:
            image = open_image(image_path).convert('RGB')
            image = np.array(image)
            faces = self._FaceDetector.find_faces(image)

            embeddings = []
            for face in faces:
                embeddings.append(self._Encoder.generate_embedding(face))

            return embeddings
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logging.error(f'Cannot create embedding for {image_path}! {e}')
            return []