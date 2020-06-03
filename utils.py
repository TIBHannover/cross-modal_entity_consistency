import json
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
    with open(path, 'r') as f:
        for line in f:
            d = json.loads(line)
            if keep_keys is not None:
                d = {k: xpath_get(d, v) for k, v in keep_keys.items()}
            data.append(d)

    if dict_key is not None:
        data = {xpath_get(x, dict_key): x for x in data}

    return data


def postprocess_text(text):
    while True:
        # Undo mistakes while removing whitespaces
        result = re.search(r'^(.*?)([\.\?\!\"\'“„])([A-Z\"\'“„])(.*?)$', text, re.DOTALL)
        if result:
            text = result.group(1) + result.group(2) + ' ' + result.group(3) + result.group(4)
        else:
            break

    while True:
        result = re.search(r'^(.*?)([a-z]|[0-9])([A-Z])([^ ])(.*?)$', text, re.DOTALL)
        if result:
            text = result.group(1) + result.group(2) + ' ' + result.group(3) + result.group(4) + result.group(5)
        else:
            break

    return text
