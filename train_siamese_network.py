import argparse
from siamese_network.provide_data import DataProvider, NLIDataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from siamese_network.siames_model import SiameseBertClassifier
from torch.optim import AdamW
from siamese_network.utils import train
import os
import logging


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ArgumentManager:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._add_arguments()

    def _add_arguments(self):
        self.parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased",
                                 help="siamese language model that fine tuned to generate NLI specific embeddings")
        self.parser.add_argument("--tokenizer_name_or_path", type=str, default=None,
                                 help="siamese language model tokenizer if it is not as same as model")
        self.parser.add_argument("--tokenizer_max_length", type=int, default=128,
                                 help="max tokens that tokenizer processed")
        self.parser.add_argument("--data", type=str, nargs="+", default=["SNLI", "MNLI", "ANLI"],
                                 help="datasets which are used to train siamese model")
        self.parser.add_argument("--train_batch_size", type=int, default=8,
                                 help="train batch size for training siamese model")
        self.parser.add_argument("--test_batch_size", type=int, default=64,
                                 help="test batch size for evaluating model at the end of each epoch")
        self.parser.add_argument("--learning_rate", type=float, default=2e-5,
                                 help="learning rate for optimizer")
        self.parser.add_argument("--dropout_rate", type=float, default=0.2,
                                 help="dropout rate for MLP part of siamese. if don't want to use set it 0")
        self.parser.add_argument("--MLP_no_layers", type=int, default=3,
                                 help="number of layers of the MLP part of siamese")
        self.parser.add_argument("--MLP_number_of_neurons", type=int, nargs="+", default=[128, 64],
                                 help="number of neurons for each layer of MLP. it must be a list with length equal to"
                                      " --MLP_no_layers - 1. last layer number of neurons is 3 because the problem is 3"
                                      " class classification")
        self.parser.add_argument("--no_unfreeze_layer", type=int, default=4,
                                 help="to avoid overfitting we just fine tune last layers. with this argument the"
                                      " number of layers fine tunes determined")
        self.parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
        self.parser.add_argument("--output_path", type=str, default="./siamese_model/",
                                 help="output directory to save model")

    def parse(self):
        return self.parser.parse_args()


if __name__ == "__main__":
    arg_manager = ArgumentManager()
    args = arg_manager.parse()
    logger.info(f"start executing")
    data_provider = DataProvider(args.data)
    data_train_premise, data_train_hypothesis, data_train_labels,\
    data_test_premise, data_test_hypothesis, data_test_label = data_provider.provide_data()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = NLIDataset(data_train_premise, data_train_hypothesis, data_train_labels, tokenizer,
                               max_length=args.tokenizer_max_length)
    test_dataset = NLIDataset(data_test_premise, data_test_hypothesis, data_test_label, tokenizer,
                              max_length=args.tokenizer_max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    model = SiameseBertClassifier(
        pretrained_model=args.model_name_or_path,
        no_unfreeze_layer=args.no_unfreeze_layer,
        mlp_number_of_neurons=args.MLP_number_of_neurons,
        dropout_rate=args.dropout_rate,
    ).to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    model = train(model, train_loader, test_loader, optimizer, device, epochs=args.epochs)

    if os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, f"{args.output_path}/checkpoint.pt")

    logger.info(f"siamese bert model saved at {args.output_path}/checkpoint.pt")




