import argparse
import os.path
from siamese_network.provide_data import DataProvider
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import nltk
nltk.download('punkt_tab')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import logging


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ArgumentManager:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._add_arguments()

    def _add_arguments(self):
        self.parser.add_argument("--data", type=str, nargs='+', default=["SNLI", "MNLI"],
                                 help="datasets used in model training")
        self.parser.add_argument("--min_freq", type=int, default=10,
                                 help="minimum frequency of a term in dataset to consider")
        self.parser.add_argument("--output_path", type=str, default="./pmi",
                                 help="output path of pmi.pkl and terms.pkl to store")

    def parse(self):
        return self.parser.parse_args()


def stem_text(text, ps):
  words = word_tokenize(text)
  stems = [ps.stem(w) for w in words]
  result = ""
  for stem in stems:
    result += stem + " "
  return result.strip()


if __name__ == "__main__":
    arg_manager = ArgumentManager()
    args = arg_manager.parse()
    data_provider = DataProvider(args.data)
    _, data_train_hypothesis, data_train_labels, _, _, _ = data_provider.provide_data()
    ps = PorterStemmer()
    data_train_hypothesis = [stem_text(text, ps) for text in data_train_hypothesis]
    vectorizer = CountVectorizer(lowercase=True, min_df=args.min_freq)
    X = vectorizer.fit_transform(data_train_hypothesis)
    terms = np.array(vectorizer.get_feature_names_out())
    labels = np.array(data_train_labels)
    total_word_count = np.asarray(X.sum(axis=0)).flatten()  # هر وازه چند بار اومده
    word_contr_count = np.asarray(X[labels == 2].sum(axis=0)).flatten()  # هر وازه چند بار تو متناقض ها اومده
    total_words = total_word_count.sum()
    total_words_contr = word_contr_count.sum()
    p_word = total_word_count / total_words
    p_word_and_contr = word_contr_count / total_words_contr
    p_contr = total_words_contr / total_words
    with np.errstate(divide='ignore'):
        pmi = np.log2(p_word_and_contr / p_word)
        pmi[np.isinf(pmi)] = 0
        pmi[pmi < 0] = 0
        pmi += 1  # change range form pmi from 1 to inf

    top_indices = np.argsort(-pmi)[:10]
    logger.info(f"top 10 terms with highest pmi")
    for idx in top_indices:
        logger.info(f"{terms[idx]}: {pmi[idx]:.4f}")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    np.save(f"{args.output_path}/terms.pkl", terms)
    logger.info(f"terms saved in {args.output_path}/terms.pkl")
    np.save(f"{args.output_path}/pmi.pkl", pmi)
    logger.info(f"pmi saved in {args.output_path}/pmi.pkl")


