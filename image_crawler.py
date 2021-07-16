from datetime import datetime
import hashlib
import os

from icrawler import ImageDownloader
from icrawler.builtin import BingImageCrawler
from six.moves.urllib.parse import urlparse


class CostumDownloader(ImageDownloader):
    def __init__(self, *args, **kwargs):
        self._entity_type = kwargs["entity_type"]
        self._entity = kwargs["entity"]
        self._root_dir = kwargs["root_dir"]
        self._engine = kwargs["engine"]
        self._license = kwargs["license"]
        self.entity_dict = []

        del kwargs["entity_type"]
        del kwargs["entity"]
        del kwargs["root_dir"]
        del kwargs["engine"]
        del kwargs["license"]

        super(CostumDownloader, self).__init__(*args, **kwargs)

    def get_filename(self, task, default_ext):
        url_path = urlparse(task["file_url"])[2]
        if "." in url_path:
            extension = url_path.split(".")[-1]
            if extension.lower() not in ["jpg", "jpeg", "png", "bmp", "tiff", "gif", "ppm", "pgm", "webp"]:
                extension = default_ext
        else:
            extension = default_ext

        # works for python 3
        m = hashlib.sha256()
        m.update(task["file_url"].encode())

        return "{}_{}.{}".format(self._engine, m.hexdigest(), extension)

    def process_meta(self, task):
        if task["success"]:
            self.entity_dict.append(
                {
                    "filename": os.path.join(self._root_dir, task["filename"]),
                    "url": task["file_url"],
                    "engine": self._engine,
                    "license": self._license,
                    "download_date": str(datetime.now()),
                }
            )

    def download(self, task, default_ext=".jpg", timeout=5, max_retry=3, overwrite=False, **kwargs):
        """Download the image and save it to the corresponding path.

        Args:
            task (dict): The task dict got from ``task_queue``.
            timeout (int): Timeout of making requests for downloading images.
            max_retry (int): the max retry times if the request fails.
            **kwargs: reserved arguments for overriding.
        """
        file_url = task["file_url"]
        task["success"] = False
        task["filename"] = None
        retry = max_retry

        if not overwrite:
            with self.lock:
                self.fetched_num += 1
                filename = self.get_filename(task, default_ext)
                if self.storage.exists(filename):
                    self.logger.info("skip downloading file %s", filename)
                    task["filename"] = filename
                    task["success"] = True
                    return
                self.fetched_num -= 1

        while retry > 0 and not self.signal.get("reach_max_num"):
            try:
                response = self.session.get(file_url, timeout=timeout)
            except Exception as e:
                self.logger.error(
                    "Exception caught when downloading file %s, " "error: %s, remaining retry times: %d",
                    file_url,
                    e,
                    retry - 1,
                )
            else:
                if self.reach_max_num():
                    self.signal.set(reach_max_num=True)
                    break
                elif response.status_code != 200:
                    self.logger.error("Response status code %d, file %s", response.status_code, file_url)
                    break
                elif not self.keep_file(task, response, **kwargs):
                    break
                with self.lock:
                    self.fetched_num += 1
                    filename = self.get_filename(task, default_ext)
                self.logger.info("image #%s\t%s", self.fetched_num, file_url)
                self.storage.write(filename, response.content)
                task["success"] = True
                task["filename"] = filename
                break
            finally:
                retry -= 1


def download_bing_images(
    entity: str, entity_type: str, download_folder: str, num_images: int, img_license: str, use_entity_type_query=False
):

    # create output folder
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # init crawler
    crawler = BingImageCrawler(
        feeder_threads=1,
        parser_threads=1,
        downloader_threads=4,
        downloader_cls=CostumDownloader,
        storage={"backend": "FileSystem", "root_dir": download_folder},
        extra_downloader_args={
            "entity_type": entity_type,
            "entity": entity,
            "root_dir": download_folder,
            "engine": "bing",
            "license": img_license,
        },
    )

    # specify search query
    if img_license == "noncommercial":
        filters = dict(type="photo", license="noncommercial")
    else:  # license == 'all':
        filters = dict(type="photo")

    if use_entity_type_query:
        keyword = entity + " " + entity_type
    else:
        keyword = entity

    # crawl images
    crawler.crawl(keyword=keyword, max_num=num_images, filters=filters)
    return crawler.downloader.entity_dict
