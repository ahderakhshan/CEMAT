import json
import nltk
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


class DataReader:
    def __init__(self, data_path):
        self.data_path = data_path

    # we have make this code for our special data type if necessary change it
    def read_documents(self):
        with open(self.data_path, "r") as f:
            data = json.load(f)
            documents = [data["hits"][i]["ABSTRACT"] for i in range(len(data["hits"]))]
        return documents

    def retrieve_sentences(self):
        documents = self.read_documents()
        all_sentences = []
        for abstract in documents:
            sentences = sent_tokenize(abstract)
            all_sentences += [s for s in sentences]
        return all_sentences

