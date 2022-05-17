import csv
import json
import os
import re


def xpath_get(mydict, path):
    elem = mydict
    try:
        for x in path.strip("/").split("/"):
            try:
                x = int(x)
                elem = elem[x]
            except ValueError:
                elem = elem.get(x)
    except:
        pass
    return elem


def read_jsonl(path, dict_key=None, keep_keys=None):
    data = []
    with open(path, "r") as f:
        for line in f:
            d = json.loads(line)
            if keep_keys is not None:
                d = {k: xpath_get(d, v) for k, v in keep_keys.items()}
            data.append(d)

    if dict_key is not None:
        data = {xpath_get(x, dict_key): x for x in data}

    return data


def read_split(fname):
    test_doc_ids = set()
    with open(fname) as csvfile:
        content = csv.reader(csvfile)
        for row in content:
            test_doc_ids.add(row[0])

    return test_doc_ids


def generate_testset(dataset, entity_type, scenes):
    anno_key = "annotation_" + entity_type
    testkey = "test_" + entity_type

    test_docs = {}
    for doc in dataset:
        # check if document contains entities of the given type
        if len(dataset[doc][testkey]["untampered"]) < 1:
            continue

        # check if the scene of the document matches the specified test setup
        if dataset[doc]["scene_label"] not in scenes:
            continue

        if anno_key in dataset[doc].keys():  # check if annotation exists
            if dataset[doc][anno_key] == 1:  # only consider positive (value=1) documents
                test_docs[doc] = dataset[doc]
        else:  # if no annotation consider all documents
            test_docs[doc] = dataset[doc]

    return test_docs


def postprocess_text(text):
    while True:
        # Undo mistakes while removing whitespaces
        result = re.search(r"^(.*?)([\.\?\!\"\'“„])([A-Z\"\'“„])(.*?)$", text, re.DOTALL)
        if result:
            text = result.group(1) + result.group(2) + " " + result.group(3) + result.group(4)
        else:
            break

    while True:
        result = re.search(r"^(.*?)([a-z]|[0-9])([A-Z])([^ ])(.*?)$", text, re.DOTALL)
        if result:
            text = result.group(1) + result.group(2) + " " + result.group(3) + result.group(4) + result.group(5)
        else:
            break

    return text


def find_news_text(text):
    results = re.match(r"^Image copyright (.*?) Image caption (.*?)\n\n(.*?)$", text, re.DOTALL)
    if results:
        image_copyright_by = results.group(1)
        image_caption = results.group(2)
        text = results.group(3)
        return image_copyright_by, image_caption, text
    else:
        return "", "", text


def allowed_image_file(image_file):
    return os.path.splitext(image_file)[1].lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
    }
