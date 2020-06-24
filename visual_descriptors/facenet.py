import math

import numpy
import os
import re
import tensorflow
from tensorflow.python.platform import gfile

from typing import List, Tuple


def prewhiten(x):
    mean = numpy.mean(x)
    std = numpy.std(x)
    std_adj = numpy.maximum(std, 1.0 / numpy.sqrt(x.size))
    y = numpy.multiply(numpy.subtract(x, mean), 1 / std_adj)
    return y


def load_model(model: os.PathLike, input_map: dict = None) -> None:
    """
    Load tensorflow model

    :param model: Folder that contains the network model
    :param input_map: Optional input mapping.
                      See: https://www.tensorflow.org/api_docs/python/tf/graph_util/import_graph_def
    """
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    # or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tensorflow.GraphDef()
            graph_def.ParseFromString(f.read())
            tensorflow.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tensorflow.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tensorflow.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir: os.PathLike) -> Tuple[str, str]:
    """
    Searches given directory for a tensorflow meta and model file.

    :param model_dir: Directory to search
    :return: Meta and model file as tuple
    """
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tensorflow.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def distance(embeddings1: List[numpy.ndarray], embeddings2: List[numpy.ndarray],
             distance_metric: int = 0) -> numpy.ndarray:
    """
    Compares embeddings1[x] to embeddings2[x] with the set metric.

    :param embeddings1: List of embeddings
    :param embeddings2: List of embeddings to compare to
    :param distance_metric: 0 for euclidean distance (default),
                            1 for cosine similarity
    :return: distances between entries
    """
    if distance_metric == 0:
        # Euclidean distance
        diff = numpy.subtract(embeddings1, embeddings2)
        dist = numpy.sum(numpy.square(diff), 1)
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dot = numpy.sum(numpy.multiply(embeddings1, embeddings2), axis=1)
        norm = numpy.linalg.norm(embeddings1, axis=1) * numpy.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = numpy.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist
