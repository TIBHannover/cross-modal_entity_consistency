import logging
import spacy


def get_spacy_annotations(text, language="en"):
    if language == "en":
        logging.info("Set language en for spaCy")
        spacy_ner = spacy.load("en_core_web_sm")
    elif language == "de":
        logging.info("Set language de for spaCy")
        spacy_ner = spacy.load("de_core_news_sm")
    else:
        logging.error(f"Unsupported language {language}. Please use [en, de]!")
        return []

    doc = spacy_ner(text)
    named_entities = []
    for ent in doc.ents:
        named_entities.append(
            {
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "cms": None,
            }
        )
    return named_entities
