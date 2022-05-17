import os
import requests
import tarfile
from tqdm import tqdm
from utils import read_jsonl

cur_dir = os.path.dirname(__file__)
out_dir = os.path.join(cur_dir, "resources")

dataset_links = read_jsonl(os.path.join(cur_dir, "dataset_links.jsonl"))

"""
Download files
"""
for x in dataset_links:
    for url in x["urls"]:
        fname = url.split("/")[-1]
        out_path = os.path.join(out_dir, x["subdir"], fname)

        # create output directory
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))

        # check if file already exists
        if os.path.isfile(out_path) or os.path.isfile(os.path.splitext(out_path)[0]):
            print(f"{fname} already exists.")
            continue

        # download file
        try:
            r = requests.get(x["base_url"] + url, stream=True)
            total_size = int(r.headers.get("content-length", 0))
            block_size = 1024
            t = tqdm(total=total_size, unit="iB", unit_scale=True, desc="Downloading " + fname)
            with open(out_path, "wb") as f:
                for data in r.iter_content(block_size):
                    t.update(len(data))
                    f.write(data)
            t.close()
        except KeyboardInterrupt:
            if os.path.isfile(out_path):
                os.remove(out_path)
            print(f"Error downloading: {fname}")
        except:
            if os.path.isfile(out_path):
                os.remove(out_path)
            print(f"Error downloading: {fname}")

"""
Untar files
"""
for x in dataset_links:
    fpath = os.path.join(out_dir, x["subdir"], x["urls"][0].split("/")[-1])

    if fpath.endswith("bin.gz"):  # fastText models
        if os.path.isfile(os.path.splitext(fpath)[0]):
            print(f"{os.path.splitext(fpath)[0]} already exists")
            continue

        print(f"gunzip {fpath}")
        os.system(f"gunzip {fpath}")
        continue

    # tar.gz
    if len(x["urls"]) > 1:
        partname, _ = os.path.splitext(fpath)
        if not os.path.exists(partname):
            print(f"Combine parts of {partname}")
            os.system(f"cat {partname}* > {partname}")

        fpath = partname

    print(f"Untar {fpath}")
    tf = tarfile.open(fpath)
    tf.extractall(path=os.path.dirname(fpath))

    # move downloaded file from old icmr'20 dataset to new ijmir'21 folder structure
    if x["name"] == "tamperednews_wordembeddings":
        src_name = os.path.join(out_dir, "wordembeddings", "word_embeddings_nouns.h5")
        dst_name = os.path.join(out_dir, "features", "tamperednews_noun_embeddings.h5")

        print(src_name)

        if os.path.isdir(os.path.dirname(src_name)):
            if os.path.isfile(src_name):
                print(f"mv {src_name} {dst_name}")
                os.system(f"mv {src_name} {dst_name}")

            print(f"rm -r {os.path.dirname(src_name)}")
            os.system(f"rm -r {os.path.dirname(src_name)}")
