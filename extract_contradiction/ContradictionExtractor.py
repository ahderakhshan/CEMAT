import os.path

import torch
import random
import torch.nn.functional as F
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import csv
from tqdm import tqdm


class ContradictionExtractor:
    def __init__(self, siamese_model, sentences, embeddings, representative_point_type, terms_pickle_path,
                 pmi_pickle_path, output_path, e_th=0.7, c_th=0.7, m=3, r=0.3, similarity_factor=1,
                 probability_factor=1, bias_factor=1, length_factor=1, number_of_extractions=None):
        self.siamese_model = siamese_model
        self.embeddings = embeddings
        self.sentences = sentences
        self.embeddings_dim = self.embeddings.shape[1]
        self.representative_point_type = representative_point_type
        assert self.representative_point_type in ["Mean", "AroundMean", "Random", "Medoids"], \
            "undefined representative point type"
        self.m = 1 if representative_point_type == "Mean" else m  # number of selected representative points
        self.r = r  # in AroundMean point type this parameter determines how many go far away from mean
        self.e_th = e_th
        self.c_th = c_th
        self.terms = np.load(terms_pickle_path, allow_pickle=True)
        self.pmi = np.load(pmi_pickle_path, allow_pickle=True)
        self.similarity_factor = similarity_factor
        self.probability_factor = probability_factor
        self.bias_factor = bias_factor
        self.length_factor = length_factor
        self.pmi_cache = {}
        self.stemmer = PorterStemmer()
        self.number_of_extractions = number_of_extractions
        self.output_path = output_path

    def save_contradictory_pairs(self):
        pairs = self._sort_pairs()

        output_directory_parts = self.output_path.split("/")[:-1]
        output_directory = ""
        for i in output_directory_parts:
            output_directory += f"{i}/"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        if self.number_of_extractions is None:
            self.number_of_extractions = len(pairs)
        else:
            self.number_of_extractions = min(self.number_of_extractions, len(pairs))
        pairs = pairs[:self.number_of_extractions]
        output_file = open(self.output_path, "w")
        output_writer = csv.writer(output_file)
        output_writer.writerow(["sentence1", "sentence2", "score"])
        for pair in pairs:
            sentence1 = self.sentences[pair[0]]
            sentence2 = self.sentences[pairs[1]]
            score = self.sentences[pairs[2]]
            output_writer.writerow([sentence1, sentence2, score])
        output_file.close()

    def _sort_pairs(self):
        all_pairs = self._get_all_pairs()
        pairs_scores = np.zeros((len(all_pairs), 3))
        pair_counter = 0
        for pair in tqdm(all_pairs):
            sentence1, sentence2, point_number = self.sentences[pair[0]], self.sentences[pair[1]], pair[2]
            similarity_score = self.compute_jacard_similarity(sentence1, sentence2) if self.similarity_factor != 0 else 0
            length_score = min(len(sentence2.split(" ")) / len(sentence1.split(" ")), 1)
            pmi_score = self.compute_pmi_score(pair[1]) if self.bias_factor != 0 else 0
            prob_score = self.compute_probability_score(pair[0], pair[1], point_number) if self.probability_factor != 0 else 0
            final_score = self.similarity_factor*similarity_score + self.length_factor*length_score +\
                          self.bias_factor*pmi_score + self.probability_factor*prob_score
            pairs_scores[pair_counter] = [pair[0], pair[1], final_score]
            pair_counter += 1
        pairs_scores = sorted(pairs_scores, key=lambda x: x[2], reverse=True)
        return pairs_scores

    def _get_all_pairs(self):
        all_pairs = set()
        point_pairs = self._get_points_pairs()
        for point_number in point_pairs.keys():
            point_entailments = point_pairs[point_number]["entailment"]
            point_contradictions = point_pairs[point_number]["contradiction"]
            for entailment in point_entailments:
                for contradiction in point_contradictions:
                    all_pairs.add((entailment, contradiction, point_number))
        return all_pairs

    def _get_points_pairs(self):
        representative_points_data_status = {}
        points_probs = self._compute_probability_with_points()
        point_counter = 0
        for prob in points_probs:
            representative_points_data_status[point_counter] = {"entailment": [], "contradiction": []}
            sentence_idx = 0
            for instance in prob:
                if instance[0] >= self.e_th:
                    representative_points_data_status[point_counter]["entailment"].append(sentence_idx)
                elif instance[2] >= self.c_th:
                    representative_points_data_status[point_counter]["contradiction"].append(sentence_idx)
                sentence_idx += 1
            point_counter += 1
        return representative_points_data_status

    def _compute_probability_with_points(self):
        representative_points = self._select_representative_points()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        all_probs = []
        for point in representative_points:
            repeated_point = point.repeat(self.embeddings.shape[0], 1).to(device)
            abs_diff = torch.abs(self.embeddings - repeated_point)
            elem_mult = self.embeddings * repeated_point
            combined = torch.cat([self.embeddings, repeated_point, abs_diff, elem_mult], dim=1)
            logits = self.siamese_model.classifier(combined)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)
        self.all_probs = all_probs
        return all_probs

    def _select_representative_points(self):
        if self.representative_point_type == "Mean":
            random_points = self._select_mean_point()
        elif self.representative_point_type == "AroundMean":
            random_points = self._select_around_mean_points()
        elif self.representative_point_type == "Random":
            random_points = self._select_random_points()
        elif self.representative_point_type == "Medoids":
            random_points = self._select_medoids_points()
        return random_points

    def _select_mean_point(self):
        random_point = []
        for i in range(self.embeddings_dim):
            value = self.embeddings[:, i].mean().item()
            random_point.append(value)
        random_points = torch.tensor([random_point])
        return random_points

    def _select_around_mean_points(self):
        random_points = []
        for i in range(self.m):
            random_point = []
            for j in range(self.embeddings_dim):
                value = self.embeddings[:, j].mean().item()
                value += random.uniform(-self.r, self.r)
                random_point.append(value)
            random_points.append(random_point)
        return torch.tensor(random_points)

    def _select_random_points(self):
        random_points = []
        for i in range(self.m):
            random_point = []
            for j in range(self.embeddings_dim):
                max_j = self.embeddings[:, j].max()
                min_j = self.embeddings[:, j].min()
                value = random.uniform(min_j, max_j)
                random_point.append(value)
            random_points.append(random_point)
        return torch.tensor(random_points)

    def _select_medoids_points(self):
        indices = torch.randperm(self.embeddings.size(0))[:self.m]
        return self.embeddings[indices]

    def compute_pmi_score(self, c):
        try:
            return self.pmi_cache[c]
        except:
            sentence = self.sentences[c]
            for word in sentence.split(" "):
                text = self.lemmatize_text(sentence)
                words = text.split(" ")
                sum_pmi = 0
                for word in words:
                    try:
                        sum_pmi += self.pmi[np.where(self.terms == word)[0]][0]
                    except:
                        sum_pmi += 1
                self.pmi_cache[c] = len(words) / sum_pmi
                return len(words) / sum_pmi

    def lemmatize_text(self, text):
        words = word_tokenize(text)
        stems = [self.stemmer.stem(w) for w in words]
        result = ""
        for stem in stems:
            result += stem + " "
        return result.strip()

    def compute_probability_score(self, e, c, point_number):
        e_prob = self.all_probs[point_number][e][0]
        c_prob = self.all_probs[point_number][c][2]
        return (e_prob + c_prob) / 2

    @classmethod
    def compute_jacard_similarity(cls, sentence1, sentence2):
        sentence1_words = set(sentence1.split(" "))
        sentence2_words = set(sentence2.split(" "))
        union = sentence1_words.union(sentence2_words)
        intersect = sentence1_words.intersection(sentence2)
        return len(union) / len(intersect)
