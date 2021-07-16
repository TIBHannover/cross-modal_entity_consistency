import csv
import os
from multiprocessing.pool import ThreadPool as Pool
from named_entity_linking.wikidata import get_entity_response, get_wikidata_entries

_CUR_DIR = os.path.dirname(__file__)

eventKG = set()
with open(os.path.join(_CUR_DIR, "..", "eventKG", "eventKG.csv"), "r") as csvfile:
    content = csv.reader(csvfile)
    for row in content:
        eventKG.add(row[0])


def fix_entity_types(linked_entities, language="en", threads=8):
    # get knowledge graph information from wikidata
    entity_list = []
    for ent in linked_entities:
        if ent["wd_id"] not in entity_list:
            entity_list.append(ent["wd_id"])

    entity_info = {}
    with Pool(threads) as p:
        for i, wd_info in enumerate(
            p.imap(lambda x: get_entity_response(x, language=language), [x for x in entity_list])
        ):
            wd_id = entity_list[i]
            entity_info[wd_id] = wd_info

    # fix entity types based on knowledge graph information
    for i in range(len(linked_entities)):
        wd_id = linked_entities[i]["wd_id"]

        is_event = False
        is_person = False
        is_location = False

        if wd_id in eventKG:
            is_event = True

        information = ["wikipedia_url", "entityDescription", "wdimage"]
        for b in entity_info[wd_id]["bindings"]:
            wd_label = b.get("entityLabel", {"value": None}).get("value")
            if wd_label is None:
                linked_entities[i]["wd_label"] = "undefined"
                continue
            linked_entities[i]["wd_label"] = wd_label

            if "instance" in b and "value" in b["instance"] and b["instance"]["value"].endswith("/Q5"):
                is_person = True

            if "coordinate" in b and "value" in b["coordinate"]:
                is_location = True

            for info_tag in information:
                if info_tag in b and "value" in b[info_tag]:
                    linked_entities[i][info_tag] = b[info_tag]["value"]
                else:
                    linked_entities[i][info_tag] = ""

        # if "wdimage" not in linked_entities[i] or linked_entities[i]["wdimage"] == "":  # set placeholder image
        #     linked_entities[i]["wdimage"] = "http://www.jennybeaumont.com/wp-content/uploads/2015/03/placeholder.gif"

        # set placeholder for card view
        linked_entities[i]["reference_images"] = []
        if "wdimage" in linked_entities[i] and linked_entities[i]["wdimage"] != "":
            linked_entities[i]["reference_images"] = [{"url": linked_entities[i]["wdimage"], "source": "wikidata"}]

        if is_location:
            linked_entities[i]["type"] = "LOCATION"
        if is_person:  # NOTE higher priority if an entity is an instance of person then it cannot be a location
            linked_entities[i]["type"] = "PERSON"
        if is_event:  # NOTE highest priority as the entity is covered by EventKG
            linked_entities[i]["type"] = "EVENT"
        if not (is_location or is_person or is_event):
            linked_entities[i]["type"] = "unknown"

    return linked_entities


def link_annotations(spacy_annotations, wikifier_annotations, threads=8):
    linked_entities = []
    with Pool(threads) as p:
        for i, ent in enumerate(p.imap(link_annotations_pool, [(x, wikifier_annotations) for x in spacy_annotations])):
            if ent is not None:
                linked_entities.append(ent)

    return linked_entities


def link_annotations_pool(args):
    POSSIBLE_SPACY_TYPES = ["PER", "PERSON", "FAC", "ORG", "GPE", "LOC", "EVENT", "MISC"]

    spacy_anno = args[0]
    wikifier_annotations = args[1]

    # skip all entities with 0 or 1 characters or not in selected spacy types
    if len(spacy_anno["text"]) < 2 or spacy_anno["type"] not in POSSIBLE_SPACY_TYPES:
        return None

    related_wikifier_entries = get_related_wikifier_entry(spacy_anno, wikifier_annotations)

    # if no valid wikifier entities were found, try to find entity based on string using <wbsearchentities>
    if len(related_wikifier_entries) < 1:
        # get wikidata id for extrated text string from spaCy NER
        entity_candidates = get_wikidata_entries(entity_string=spacy_anno["text"], limit_entities=1, language="en")

        # if also no match continue with next entity
        if len(entity_candidates) < 1:
            return None

        # take the first entry in wbsearchentities (most likely one)
        entity_candidate = {
            **{
                "wd_id": entity_candidates[0]["id"],
                "wd_label": entity_candidates[0]["label"],
                "disambiguation": "wbsearchentities",
            },
            **spacy_anno,
        }
    else:
        highest_PR = -1
        best_wikifier_candidate = related_wikifier_entries[0]
        for related_wikifier_entry in related_wikifier_entries:
            # print(related_wikifier_entry['title'], related_wikifier_entry['pageRank_occurence'])
            if related_wikifier_entry["pageRank_occurence"] > highest_PR:
                best_wikifier_candidate = related_wikifier_entry
                highest_PR = related_wikifier_entry["pageRank_occurence"]

        entity_candidate = {
            **{
                "wd_id": best_wikifier_candidate["wikiDataItemId"],
                "wd_label": best_wikifier_candidate["secTitle"],
                "disambiguation": "wikifier",
            },
            **spacy_anno,
        }

    return entity_candidate


def get_related_wikifier_entry(spacy_anno, wikifier_annotations, char_tolerance=2, threshold=1e-4):
    # loop through entities found by wikifier
    aligned_candidates = []
    for wikifier_entity in wikifier_annotations["annotations"]:
        if "secTitle" not in wikifier_entity.keys() or "wikiDataItemId" not in wikifier_entity.keys():
            continue

        wikifier_entity_occurences = wikifier_entity["support"]

        # loop through all occurences of a given entity recognized by wikifier
        for wikifier_entity_occurence in wikifier_entity_occurences:

            if wikifier_entity_occurence["chFrom"] < spacy_anno["start"] - char_tolerance:
                continue

            if wikifier_entity_occurence["chTo"] > spacy_anno["end"] + char_tolerance:
                continue

            # apply very low threshold to get rid of annotation with very low confidence
            if wikifier_entity_occurence["pageRank"] < threshold:
                continue

            aligned_candidates.append(
                {**wikifier_entity, **{"pageRank_occurence": wikifier_entity_occurence["pageRank"]}}
            )

    return aligned_candidates
