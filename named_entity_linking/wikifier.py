import json
import logging
import urllib.parse
from urllib.request import Request


def get_wikifier_annotations(text, language, wikifier_key):
    threshold = 1.0
    language = language
    wikiDataClasses = "false"
    wikiDataClassIds = "true"
    includeCosines = "false"

    try:
        data = urllib.parse.urlencode(
            [
                ("text", text),
                ("lang", language),
                ("userKey", wikifier_key),
                ("pageRankSqThreshold", "%g" % threshold),
                ("applyPageRankSqThreshold", "true"),
                ("nTopDfValuesToIgnore", "200"),
                ("nWordsToIgnoreFromList", "200"),
                ("wikiDataClasses", wikiDataClasses),
                ("wikiDataClassIds", wikiDataClassIds),
                ("support", "true"),
                ("ranges", "false"),
                ("includeCosines", includeCosines),
                ("maxMentionEntropy", "3"),
            ]
        )

        req = urllib.request.Request("http://wikifier.org/annotate-article", data=data.encode("utf8"), method="POST")
        with urllib.request.urlopen(req, timeout=60) as f:
            response = f.read()
            response = json.loads(response.decode("utf8"))
            if "annotations" in response:
                return {"processed": True, "annotations": response["annotations"]}
            else:
                logging.error(f"No valid response: {response}")
                return {"processed": False, "annotations": []}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logging.error(f"No reponse from Wikifier: {e}")
        return {"processed": False, "annotations": []}
