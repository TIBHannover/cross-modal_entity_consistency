import argparse
import h5py as h5py
import logging
import os
import sys

from word_embedder import WordEmbedder
import utils


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-vv', '--debug', action='store_true', help='debug output')

    parser.add_argument('-d', '--dataset', type=str, required=True, help='path to dataset.jsonl containing news texts')
    parser.add_argument('-f', '--fasttext', type=str, required=True, help='FastText bin folder')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to output directory')

    parser.add_argument('--tokens', nargs='+', type=str, default=["NOUN"], required=False, help='tokens to process')
    args = parser.parse_args()
    return args


def main():
    # load arguments
    args = parse_args()

    # define logging level and format
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=level)

    # create model depending on model type
    logging.info('Init WordEmbedding ...')
    if "tamperednews" in args.dataset:
        language = "en"
    else:  # news400
        language = "de"
    we = WordEmbedder(fasttext_bin_folder=args.fasttext, language=language, token_types=args.tokens)

    # create output dir
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    # read dataset
    dataset = utils.read_jsonl(args.dataset, dict_key='id')

    # check if output file already exists
    logging.info('Generate word embeddings ...')
    if os.path.isfile(args.output):
        mode = 'r+'
    else:
        mode = 'a'

    # save embeddings to h5
    with h5py.File(args.output, mode) as output_file:
        cnt_docs = 0

        for document in dataset.values():
            cnt_docs += 1
            if cnt_docs % 100 == 0:
                logging.info(f'{cnt_docs} documents processed')

            if document['id'] in output_file:
                logging.debug(f"{document['id']} already processed")
                continue

            if "text" not in document.keys():
                output_file[document['id']] = []
                continue

            output_file[document['id']] = we.generate_embeddings(document["text"])

    return 0


if __name__ == '__main__':
    sys.exit(main())
