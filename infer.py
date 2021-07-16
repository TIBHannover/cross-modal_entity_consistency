import argparse
import logging
import sys
import yaml

from utils import allowed_image_file
from newsanalyzer import NewsAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference script to quantify the cross-modal consistency of entities in image-text pairs."
    )

    # required arguments
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config file")
    parser.add_argument("-i", "--image", type=str, required=True, help="Path to input image")
    parser.add_argument("-t", "--text", type=str, required=True, help="Path to input txt file")
    parser.add_argument(
        "-w",
        "--wikifier_key",
        type=str,
        required=True,
        help="Your Wikifier key from http://www.wikifier.org/register.html",
    )

    # optional arguments
    parser.add_argument("-v", "--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        required=False,
        default="en",
        choices=["en", "de"],
        help="Language of the input text",
    )

    args = parser.parse_args()
    return args


def main():
    # load arguments
    args = parse_args()

    # define logging level and format
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # read txt file
    with open(args.text, "r") as txtfile:
        text = txtfile.read()

    logging.info("Input text: ")
    logging.info(text)

    # check image input
    if not allowed_image_file(args.image):
        logging.error("Image extension unknown. Exiting ...")
        return 0

    # initialize newsanalyzer
    na = NewsAnalyzer(wikifier_key=args.wikifier_key, config=config)
    cms, entities_cms = na.get_entity_cms(image_file=args.image, text=text, language=args.language)

    # output CMI results
    unique_entities = set()
    logging.info("#### CMS for individual entities")
    for entity in entities_cms:
        logging.debug(entity)
        if entity["wd_id"] not in unique_entities and entity["type"] in ["PERSON", "LOCATION", "EVENT"]:
            logging.info(f"{entity['type']} - {entity['wd_label']} ({entity['wd_id']}): {entity['cms']}")
            unique_entities.add(entity["wd_id"])

    logging.info("#### CMS for the whole document")
    logging.info(f"CMPS: {cms['PERSON']}")
    logging.info(f"CMLS: {cms['LOCATION']}")
    logging.info(f"CMES: {cms['EVENT']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())