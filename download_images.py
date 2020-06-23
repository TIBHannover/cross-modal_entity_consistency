import argparse
import cv2
import imghdr
import logging
import multiprocessing
import numpy as np
import os
import shutil
import sys
import time
import urllib.request
import urllib.parse
import urllib.error

# own imports
import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Downloader for images")
    parser.add_argument("-vv", "--debug", action="store_true", help="debug output")

    parser.add_argument("-i", "--input", type=str, required=True, help="path to input.jsonl")
    parser.add_argument("-o", "--output", type=str, required=True, help="path to output directory")
    parser.add_argument("--type", type=str, required=True, choices=["news", "entity"], help="specify image type")

    parser.add_argument("-s", "--maxsize", type=int, required=False, help="Max. number of pixels for smaller dimension")
    parser.add_argument("-t", "--threads", type=int, default=8, required=False, help="number of downloader threads")

    args = parser.parse_args()
    return args


def download_news_images(args):
    outfile, url, maxsize = args
    if "identifier" in url:
        url = url["identifier"]

    if os.path.isfile(outfile):
        logging.info(f"Image already exists: {url}")
        return None

    logging.info(f"Downloading image from: {url}")

    try:
        request = urllib.request.Request(
                url=url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_3)"
                        " AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/48.0.2564.116 Safari/537.36"
                    )
                },
            )

        with urllib.request.urlopen(request, timeout=10) as response:
            img = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
            if maxsize is not None:
                scale = max(img.shape[0:2]) / maxsize
                if scale > 1:  # only allow downscaling
                    img = cv2.resize(img, (int(img.shape[1] / scale + 0.5), int(img.shape[0] / scale + 0.5)))

            cv2.imwrite(outfile, img)
            image_type = imghdr.what(outfile)
            if image_type == 'webp':
                fname, ext = os.path.splitext(outfile)
                shutil.move(outfile, fname + '.webp')
            return True
    except urllib.error.HTTPError as err:
        logging.error(str(err.reason))
        time.sleep(5.0)
    except urllib.error.URLError as err:
        logging.error(str(err.reason))
        time.sleep(5.0)
    except KeyboardInterrupt:
        raise

    return None


def main():
    # load arguments
    args = parse_args()

    # define logging level and format
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)

    # read urls from input file
    if args.type == "news":
        dataset = utils.read_jsonl(args.input, dict_key="id")
        if not os.path.exists(args.output):
            os.makedirs(args.output)
    else:
        dataset = utils.read_jsonl(args.input, dict_key="wd_id")

    image_urls = []
    for doc in dataset.values():
        if args.type == "news":
            outfile = os.path.join(args.output, doc["id"] + ".jpg")
            image_urls.append([outfile, doc["image_url"], args.maxsize])
        else:
            for image in doc["image_urls"]:
                outfile = os.path.join(args.output, doc["wd_id"], image["filename"])

                if not os.path.exists(os.path.dirname(outfile)):
                    os.makedirs(os.path.dirname(outfile))

                image_urls.append([outfile, image["url"], args.maxsize])

    # download images
    with multiprocessing.Pool(args.threads) as p:
        p.map(download_news_images, image_urls)

    return 0


if __name__ == "__main__":
    sys.exit(main())
