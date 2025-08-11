from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataProvider:
    def __init__(self, datasets):
        self.datasets = datasets

    def provide_data(self):
        all_train_premises, all_train_hypothesis, all_train_labels,\
        all_test_premises, all_test_hypothesis, all_test_lables = [], [], [], [], [], []
        if "SNLI" in self.datasets:
            snli_train_premises, snli_train_hypothesis, snli_train_labels,\
            snli_test_premises, snli_test_hypothesis, snli_test_labels = self._provide_snli()
            all_train_premises += snli_train_premises
            all_train_hypothesis += snli_train_hypothesis
            all_train_labels += snli_train_labels
            all_test_premises += snli_test_premises
            all_test_hypothesis += snli_test_hypothesis
            all_test_lables += snli_test_labels
            logger.info("snli dataset loaded")
        if "MNLI" in self.datasets:
            mnli_train_premises, mnli_train_hypothesis, mnli_train_labels,\
            mnli_test_premises, mnli_test_hypothesis, mnli_test_labels = self._provide_mnli()
            all_train_premises += mnli_train_premises
            all_train_hypothesis += mnli_train_hypothesis
            all_train_labels += mnli_train_labels
            all_test_premises += mnli_test_premises
            all_test_hypothesis += mnli_test_hypothesis
            all_test_lables += mnli_test_labels
            logger.info("mnli dataset loaded")
        if "ANLI" in self.datasets:
            anli_train_premises, anli_train_hypothesis, anli_train_labels,\
            anli_test_premises, anli_test_hypothesis, anli_test_labels = self._provide_anli()
            all_train_premises += anli_train_premises
            all_train_hypothesis += anli_train_hypothesis
            all_train_labels += anli_train_labels
            all_test_premises += anli_test_premises
            all_test_hypothesis += anli_test_hypothesis
            all_test_lables += anli_test_labels
            logger.info("anli dataset loaded")
        logger.info(f"datasets {self.datasets} loaded")
        return all_train_premises, all_train_hypothesis, all_train_labels, all_test_premises, all_test_hypothesis, all_test_lables

    def _provide_snli(self):
        snli_dataset = load_dataset("stanfordnlp/snli")
        snli_train_dataset = snli_dataset["train"]
        snli_test_dataset = snli_dataset["test"]

        snli_train_dataset = snli_train_dataset.filter(lambda example: example["label"] != -1)
        snli_test_dataset = snli_test_dataset.filter(lambda example: example["label"] != -1)

        snli_train_premises = list(snli_train_dataset["premise"])
        snli_train_hypothesis = list(snli_train_dataset["hypothesis"])
        snli_train_labels = list(snli_train_dataset["label"])

        snli_test_premises = list(snli_test_dataset["premise"])
        snli_test_hypothesis = list(snli_test_dataset["hypothesis"])
        snli_test_labels = list(snli_test_dataset["label"])

        return snli_train_premises, snli_train_hypothesis, snli_train_labels,\
               snli_test_premises, snli_test_hypothesis, snli_test_labels

    def _provide_mnli(self):
        mnli_dataset = load_dataset("nyu-mll/multi_nli")
        mnli_train_dataset = mnli_dataset["train"]
        mnli_test_dataset = mnli_dataset["validation_mismatched"] # it can be matched split instead of mismatched but we want to evaluate our model in hard scenario
        mnli_train_dataset = mnli_train_dataset.filter(lambda example: example["label"] != -1)
        mnli_test_dataset = mnli_test_dataset.filter(lambda example: example["label"] != -1)
        mnli_train_premises = list(mnli_train_dataset["premise"])
        mnli_train_hypothesis = list(mnli_train_dataset["hypothesis"])
        mnli_train_labels = list(mnli_train_dataset["label"])

        mnli_test_premises = list(mnli_test_dataset["premise"])
        mnli_test_hypothesis = list(mnli_test_dataset["hypothesis"])
        mnli_test_labels = list(mnli_test_dataset["label"])

        return mnli_train_premises, mnli_train_hypothesis, mnli_train_labels,\
               mnli_test_premises, mnli_test_hypothesis, mnli_test_labels

    def _provide_anli(self):
        anli_dataset = load_dataset("facebook/anli")
        anli_dataset1 = anli_dataset["train_r1"]
        anli_dataset2 = anli_dataset["train_r2"]
        anli_dataset3 = anli_dataset["train_r3"]
        anli_test_dataset1 = anli_dataset["test_r1"]
        anli_test_dataset2 = anli_dataset["test_r2"]
        anli_test_dataset3 = anli_dataset["test_r3"]

        anli_train_premises = list(anli_dataset1["premise"]) + list(anli_dataset2["premise"]) + list(
            anli_dataset3["premise"])
        anli_train_hypothesis = list(anli_dataset1["hypothesis"]) + list(anli_dataset2["hypothesis"]) + list(
            anli_dataset3["hypothesis"])
        anli_train_labels = list(anli_dataset1["label"]) + list(anli_dataset2["label"]) + list(anli_dataset3["label"])

        anli_test_premises = list(anli_test_dataset1["premise"]) + list(anli_dataset2["premise"]) + list(
            anli_test_dataset3["premise"])
        anli_test_hypothesis = list(anli_test_dataset1["hypothesis"]) + list(anli_test_dataset2["hypothesis"]) + list(
            anli_test_dataset3["hypothesis"])
        anli_test_labels = list(anli_test_dataset1["label"]) + list(anli_test_dataset2["label"]) + list(
            anli_test_dataset3["label"])
        return anli_train_premises, anli_train_hypothesis, anli_train_labels,\
               anli_test_premises, anli_test_hypothesis, anli_test_labels


class NLIDataset(Dataset):
    def __init__(self, premises, hypotheses, labels, tokenizer, max_length=128):
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc_a = self.tokenizer(
            self.premises[idx],
            padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )
        enc_b = self.tokenizer(
            self.hypotheses[idx],
            padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )
        return {
            'input_ids_a': enc_a['input_ids'].squeeze(0),
            'attention_mask_a': enc_a['attention_mask'].squeeze(0),
            'input_ids_b': enc_b['input_ids'].squeeze(0),
            'attention_mask_b': enc_b['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

