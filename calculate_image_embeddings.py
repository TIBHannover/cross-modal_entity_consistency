import absl.logging
import argparse
import h5py
import logging
import numpy as np
import os
import sys
import utils
from visual_descriptors.location_embedding import GeoEstimator
from visual_descriptors.person_embedding import FacialFeatureExtractor
from visual_descriptors.scene_embedding import SceneClassificator


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-v', '--info', action='store_true', help='info output')
    parser.add_argument('-vv', '--debug', action='store_true', help='debug output')
    
    parser.add_argument('-i', '--input', type=str, required=True, help='path to input.jsonl')
    parser.add_argument('-d', '--directory', type=str, required=True, help='input folder containing the image data')
    parser.add_argument('-t', '--type', type=str, required=True, choices=['news', 'entity'], help='specify image type')

    parser.add_argument('-o', '--output', type=str, required=True, help='Path to output directory')

    parser.add_argument('-m', '--model', type=str, required=True, help='Path to model directory')

    parser.add_argument('--logits', action='store_true', help='set to get scene probabilities for places365')
    parser.add_argument('--use_cpu', action='store_true', help='set to run script on cpu')
    args = parser.parse_args()
    return args


def find_file(filename, possible_file_extensions=['.jpg', '.png', '.webp']):
    fname = None
    for ext in possible_file_extensions:
        if os.path.exists(filename + ext):
            fname = filename + ext

    return fname


def main():
    # load arguments
    args = parse_args()

    # define logging level and format
    level = logging.INFO
    absl.logging.set_stderrthreshold('info')
    absl.logging.set_verbosity('info')
    if args.debug:
        absl.logging.set_stderrthreshold('debug')
        absl.logging.set_verbosity('debug')
        level = logging.DEBUG

    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=level)

    # check output file and create folder
    if not args.output.endswith('.h5'):
        logging.error('Output file should end with .h5. Extiting ...')
        return 0

    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    if os.path.isfile(args.output):
        mode = 'r+'
    else:
        mode = 'a'

    # create model depending on model type
    FE = None
    if 'facenet' in args.model:
        logging.info('Building network for embedding based on person verification ...')
        FE = FacialFeatureExtractor(model_path=args.model)
    elif 'location' in args.model:
        logging.info('Building network for embedding based on geolocation estimation ...')
        FE = GeoEstimator(model_path=args.model, use_cpu=args.use_cpu)
    elif 'scene' in args.model:
        logging.info('Building network for embedding based on scene classification ...')
        FE = SceneClassificator(model_path=args.model)

    if not FE:
        logging.error('Unknown model. Exiting ...')
        return 0

    if 'scene' not in args.model and args.logits:
        logging.error('Please specify a scene classification model to create scene logits.')
        return 0

    # read dataset
    if args.type == 'news':
        dataset = utils.read_jsonl(args.input, dict_key='id')
    else:
        dataset = utils.read_jsonl(args.input, dict_key='wd_id')

    logging.info(f'{len(dataset.keys())} dataset entries to process')

    # create embeddings
    with h5py.File(args.output, mode) as output_file:
        for entry in dataset.values():
            images = []
            if args.type == 'news':
                fname = os.path.join(args.directory, entry['id'])
                images.append({'fname': fname, 'search_engine': None})
            else:  # entity
                for image in entry['image_urls']:
                    fname, ext = os.path.splitext(image['filename'])
                    fname = os.path.join(args.directory, entry['wd_id'], fname)
                    images.append({'fname': fname, 'search_engine': image['search_engine']})

            for image in images:
                image_file = find_file(image['fname'])
                if not image_file:
                    logging.info(f"Cannot find image file {image['fname']}.jpg")
                    continue

                if args.type == 'news':
                    h5_key = entry['id']
                else:  # entity
                    h5_key = f"{entry['wd_id']}/{image['search_engine']}/{os.path.basename(image['fname'])}"

                if h5_key in output_file:
                    logging.info(f'Embedding for {h5_key} already computed ...')
                    continue

                logging.info(f'Generate embedding for {h5_key} ...')

                img_embeddings = []
                if args.logits:
                    img_emb = FE.get_logits(image_file)
                else:
                    img_emb = FE.get_img_embedding(image_file)

                for e in img_emb:
                    img_embeddings.append(e)

                if len(img_embeddings) == 0:
                    logging.debug(f'No embedding found for {h5_key} ...')
                    output_file[h5_key] = []
                else:
                    # convert to np array and store to output file
                    id_img_embs = np.asarray(img_embeddings, dtype=np.float32)
                    output_file[h5_key] = id_img_embs

    return 0


if __name__ == '__main__':
    sys.exit(main())
