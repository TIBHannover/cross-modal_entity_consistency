import fasttext
import logging
import numpy as np
import os
import spacy


class WordEmbedder:
    def __init__(self, fasttext_bin_folder, token_types=["NOUN"], language="en"):
        logging.info(f"Loading NER model: {language} ...")
        if language == "en":
            self._nlp = spacy.load("en_core_web_sm")
        elif language == "de":
            self._nlp = spacy.load("de_core_news_sm")

        logging.info(f"Loading word2vec model: cc.{language}.300.bin ...")
        self._word2vec = fasttext.load_model(os.path.join(fasttext_bin_folder, "cc." + language + ".300.bin"))
        self._token_types = token_types

    def _preprocess(self, text):
        """Tokenize text, remove stopwords and punctuation."""
        doc = self._nlp(text)

        if self._token_types is None:
            tokens = [token.text for token in doc]
        else:
            tokens = [token.text for token in doc if (token.pos_ in self._token_types)]
        return tokens

    def generate_embeddings(self, text):
        tokens = self._preprocess(text)
        word_embeddings = []
        for token in tokens:
            word_embeddings.append(self._word2vec.get_word_vector(token))

        if len(word_embeddings) == 0:
            return []

        return np.asarray(np.stack(word_embeddings), dtype=np.float32)
