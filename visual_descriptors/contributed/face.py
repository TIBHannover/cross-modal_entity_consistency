# coding=utf-8
"""Face Detection and Recognition"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# This is the work of David Sandberg and shanren7 remodelled into a
# high level container. It's an attempt to simplify the use of such
# technology and provide an easy to use facial recognition package.
#
# https://github.com/davidsandberg/facenet
# https://github.com/shanren7/real_time_face_recognition
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy
import os
import tensorflow
from scipy import misc
import sys

from PIL import Image

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CUR_DIR, '..'))
sys.path.append(os.path.join(CUR_DIR, '..', 'align'))

import facenet
import detect_face


class Face:

    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None


class Encoder:

    def __init__(self, facenet_model_checkpoint):
        self.sess = tensorflow.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, face: Face) -> numpy.ndarray:
        """
        Calculates the embedding for the given face

        :param face: A face object representing a face
        :return: The embedding for the given face
        """
        # Get input and output tensors
        images_placeholder = tensorflow.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tensorflow.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tensorflow.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]


class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32, gpu_memory_fraction=0.6):
        self.gpu_memory_fraction = gpu_memory_fraction
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tensorflow.Graph().as_default():
            gpu_options = tensorflow.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
            sess = tensorflow.Session(
                config=tensorflow.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        faces = []

        bounding_boxes, _ = detect_face.detect_face(image, self.minsize, self.pnet, self.rnet, self.onet,
                                                    self.threshold, self.factor)
        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = numpy.zeros(4, dtype=numpy.int32)

            img_size = numpy.asarray(image.shape)[0:2]
            face.bounding_box[0] = numpy.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = numpy.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = numpy.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = numpy.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]

            face.image = numpy.array(
                Image.fromarray(cropped).resize((self.face_crop_size, self.face_crop_size), Image.BILINEAR))

            faces.append(face)

        return faces
