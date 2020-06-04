import os
import requests
import tarfile
from tqdm import tqdm


cur_dir = os.path.dirname(__file__)
out_dir = os.path.join(cur_dir, 'resources_dtest')

base_url_news400 = "https://data.uni-hannover.de/dataset/c729ffd9-8be1-49a9-8c43-dab2f8a87753/resource/"
base_url_tamperednews = "https://data.uni-hannover.de/dataset/074ca5d5-02ce-4b1e-b90b-a6c0f1dd92f5/resource/"
urls = [
    # news400
    'af85e355-88e1-4d4d-91ec-785e1801dd47/download/news400.tar.gz',
    # news400 - features
    '23365241-435b-4dda-921c-09bd5f908010/download/news400_features.tar.gz',
    # news400 - wordembeddings
    '5500907c-2e5c-4967-a99d-4d64b6cde975/download/news400_wordembeddings.tar.gz',
    # tamperednews
    '5ba969cf-cad5-400e-bc8e-6646051c4fe4/download/tamperednews.tar.gz',
    # tamperednews - features
    '4a013a13-2bde-46c9-83df-a45f8c0550c3/download/tamperednews_features.tar.gz.partaa',
    'ec437bb3-db2a-4928-b198-bb1958c2ae19/download/tamperednews_features.tar.gz.partab',
    'd29fdb59-d73d-46b7-b840-06ba9561afc9/download/tamperednews_features.tar.gz.partac',
    '26a3a516-087a-410c-9c18-9a4c76257d5b/download/tamperednews_features.tar.gz.partad',
    '9cf148e3-1e1b-437d-9288-93b9261209a7/download/tamperednews_features.tar.gz.partae',
    '5248202b-5167-4257-9c0f-c323a859c158/download/tamperednews_features.tar.gz.partaf',
    # tamperednews - wordembeddings
    'a32f10ef-1a1d-4798-838b-f886890c618e/download/tamperednews_wordembeddings.tar.gz.partaa',
    'be0e18bb-57ca-4db7-afea-a24f9e349ba5/download/tamperednews_wordembeddings.tar.gz.partab',
    '0dc39cde-f35f-482e-af9b-1fe34de838a4/download/tamperednews_wordembeddings.tar.gz.partac',
    '0bb637d7-eaef-4cec-9b33-a56ac2468b28/download/tamperednews_wordembeddings.tar.gz.partad',
    'd250769c-9f40-443e-84bb-65286d6af1c6/download/tamperednews_wordembeddings.tar.gz.partae',
    '5f20f5f0-705d-4e7e-8b0b-a19dec2512ad/download/tamperednews_wordembeddings.tar.gz.partaf',
    '46b7da3b-c781-48a2-ac23-9a0070552298/download/tamperednews_wordembeddings.tar.gz.partag',
    'c0d338af-4a2a-46d8-88cd-38f8c94bd70d/download/tamperednews_wordembeddings.tar.gz.partah',
    'cc53733c-1b32-44a6-8602-05a7dfd94860/download/tamperednews_wordembeddings.tar.gz.partai',
    '5194a498-764c-41f9-9a2d-61b66891105d/download/tamperednews_wordembeddings.tar.gz.partaj',
    '5b560ddc-3c57-4107-b4f0-58e54e3be02b/download/tamperednews_wordembeddings.tar.gz.partak',
    '369817b2-95af-4965-a1de-40d3af00da95/download/tamperednews_wordembeddings.tar.gz.partal',
    'e5cb6e6d-1c02-4d66-84a0-2b3c2033c0b4/download/tamperednews_wordembeddings.tar.gz.partam',
    'b37e9006-f035-466f-a9f6-0e432d9a6a00/download/tamperednews_wordembeddings.tar.gz.partan',
    'bff1eaff-aa67-4d46-8e3e-3c66b63d7c23/download/tamperednews_wordembeddings.tar.gz.partao',
    'bf72bdc5-609b-45f5-91c8-0c884005c044/download/tamperednews_wordembeddings.tar.gz.partap',
    # fasttext
    'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz',
    'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.bin.gz'
]

fnames = []
for url in urls:
    fname = url.split('/')[-1]
    if fname.startswith("tamperednews"):
        subdir = "tamperednews"
        url = base_url_tamperednews + url
    elif fname.startswith("news400"):  # news400
        subdir = "news400"
        url = base_url_news400 + url
    else:
        subdir = "fasttext"

    out_path = os.path.join(out_dir, subdir, fname)
    fnames.append(out_path)
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    if os.path.isfile(out_path):
        print(f'{fname} already exists.')
        continue

    # download file
    try:
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024
        t = tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading " + fname)
        with open(out_path, 'wb') as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

    except KeyboardInterrupt:
        os.remove(out_path)
        print()
    except:
        os.remove(out_path)
        print()

parts = []
for fname in fnames:
    if fname.endswith("bin.gz"):  # fastText models
        continue

    partname, ext = os.path.splitext(fname)

    if ext.startswith(".part"):
        if partname in parts:
            continue
        else:
            parts.append(partname)
            if not os.path.exists(partname):
                print(f"Combine parts of {partname}")
                os.system(f"cat {partname}* > {partname}")

            print(f"Untar {partname}")
            tf = tarfile.open(partname)
            tf.extractall(path=os.path.dirname(partname))
    else:
        print(f"Untar {fname}")
        tf = tarfile.open(fname)
        tf.extractall(path=os.path.dirname(fname))

