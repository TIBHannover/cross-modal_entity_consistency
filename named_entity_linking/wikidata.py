import logging
import requests
import traceback
import re
from string import Template


def get_entity_response(wikidata_id, language="en"):
    query = Template(
        """
            prefix schema: <http://schema.org/>
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            SELECT ?entity ?entityLabel ?entityDescription ?instance ?coordinate ?wikipedia_url ?wdimage
            WHERE {
              VALUES (?entity) {(wd:$wd_id)}
              OPTIONAL { ?entity wdt:P31 ?instance . }
              OPTIONAL { ?entity wdt:P625 ?coordinate . }
              OPTIONAL { ?entity wdt:P18 ?wdimage . }
              OPTIONAL {
                ?wikipedia_url schema:about ?entity .
                ?wikipedia_url schema:inLanguage "$lang" . 
                ?wikipedia_url schema:isPartOf <https://$lang.wikipedia.org/> .
              }
              SERVICE wikibase:label {bd:serviceParam wikibase:language "$lang" .}
            }"""
    )
    q = query.substitute(wd_id=wikidata_id, lang=language)
    # logging.info("######################")
    # logging.info(q)
    res = get_response("https://query.wikidata.org/sparql", params={"format": "json", "query": q})
    # logging.info(res)
    # logging.info("######################")
    if res:
        return res["results"]
    else:
        return {"bindings": []}


def link_dbpedia_to_wikidata(ressource_name):

    query = (
        """
        select ?wikidataID where {
                dbr:%s owl:sameAs ?wikidataID.
        }
        """
        % ressource_name
    )
    try:
        res = get_response(
            "https://dbpedia.org/sparql",
            params={
                "default-graph-uri": "http://dbpedia.org",
                "format": "application/sparql-results+json",
                "query": query,
            },
        )
        for x in res["results"]["bindings"]:
            match = re.match(r".*?www\.wikidata\.org/entity/(.*?)$", x["wikidataID"]["value"])
            if match:
                return match.group(1)
    except Exception as e:
        logging.warning(e)
        logging.warning(traceback.format_exc())
    return None


def get_wikidata_entries(entity_string, limit_entities=7, language="en"):
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": language,
        "search": entity_string,
        "limit": limit_entities,
    }
    response = get_response("https://www.wikidata.org/w/api.php", params=params)
    if response:
        return response["search"]
    else:
        return []


def get_response(url, params):
    i = 0
    try:
        r = requests.get(url, params=params, headers={"User-agent": "your bot 0.1"})

        return r.json()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logging.warning(e)
        logging.warning(traceback.format_exc())
        logging.error(f"Got no response from wikidata. Retry {i}")  # TODO include reason r
        return {}
