import argparse
import logging
import nltk
from extract_contradiction.DataReader import DataReader
from extract_contradiction.ContradictionExtractor import ContradictionExtractor
nltk.download('punkt')
nltk.download('punkt_tab')
from siamese_network.siames_model import SiameseBertClassifier
import torch
from transformers import AutoTokenizer


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ArgumentManager:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._add_arguments()

    def _add_arguments(self):
        self.parser.add_argument("--data_path", type=str, default="./pubmed_abstracts/Coma.json",
                                 help="path to sentences for extracting contradictory pairs."
                                      " change DataReader read_documents function based on your data type")
        self.parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/checkpoint.pt",
                                 help="siamese language model that fine tuned to generate NLI specific embeddings")
        self.parser.add_argument("--tokenizer_name_or_path", type=str, default=None,
                                 help="siamese language model tokenizer if it is not as same as model")
        self.parser.add_argument("--tokenizer_max_length", type=int, default=128,
                                 help="max tokens that tokenizer processed")
        self.parser.add_argument("--MLP_no_layers", type=int, default=3,
                                 help="number of layers of the MLP part of siamese")
        self.parser.add_argument("--MLP_number_of_neurons", type=int, nargs="+", default=[128, 64],
                                 help="number of neurons for each layer of MLP. it must be a list with length equal to"
                                      " --MLP_no_layers - 1. last layer number of neurons is 3 because the problem is 3"
                                      " class classification")
        self.parser.add_argument("--output_file_path", type=str, default="./results/Coma.csv",
                                 help="output path to csv file  to save results")
        self.parser.add_argument("--representative_point_type", type=str, default="Mean",
                                 help="can be one of Mean, AroundMean, Random, Medoids")
        self.parser.add_argument("--terms_pickle_path", type=str, default="./terms.pkl",
                                 help="when running compute_pmi it store terms.pkl set the path of terms.pkl")
        self.parser.add_argument("--pmi_pickle_path", type=str, default="./pmi.pkl",
                                 help="when running compute_pmi it store pmi.pkl set the path of pmi.pkl")
        self.parser.add_argument("--e_th", type=float, default=0.7, help="e_th parameter of Method")
        self.parser.add_argument("--c_th", type=float, default=0.7, help="c_th parameter of Method")
        self.parser.add_argument("--m", type=int, default=3, help="number of representative points")
        self.parser.add_argument("--similarity_factor", type=float, default=1,
                                 help="similarity factor in retrieving pairs")
        self.parser.add_argument("--probability_factor", type=float, default=1,
                                 help="probability factor in retrieving pairs")
        self.parser.add_argument("--bias_factor", type=float, default=1,
                                 help="bias factor in retrieving pairs")
        self.parser.add_argument("--length_factor", type=float, default=1,
                                 help="length factor in retrieving pairs")
        self.parser.add_argument("--number_of_extractions", type=int, default=None,
                                 help="number of retrieved contradictory pairs to store in the file."
                                      " set None to store all")
    def parse(self):
        return self.parser.parse_args()


if __name__ == "__main__":
    arg_manager = ArgumentManager()
    args = arg_manager.parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"start extracting contradictions")
    data_reader = DataReader(args.data_path)
    sentences = data_reader.retrieve_sentences()
    logger.info(f"{len(sentences)} retrieved from input data path")
    siamese_model = SiameseBertClassifier(
        no_unfreeze_layer=args.no_unfreeze_layer,
        mlp_number_of_neurons=args.MLP_number_of_neurons,
    )
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    siamese_model.load_state_dict(checkpoint['model_state_dict'])
    siamese_model.eval()
    siamese_model.to(device)
    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    logger.info("model loaded successfully")
    logger.info("start embedding sentences")
    embeddings = siamese_model.embedd_sentences(
        sentences=sentences,
        tokenizer=tokenizer,
        max_length=args.tokenizer_max_length
    )
    contradiction_extractor = ContradictionExtractor(
        siamese_model=siamese_model,
        sentences=sentences,
        embeddings=embeddings,
        representative_point_type=args.representative_point_type,
        terms_pickle_path=args.terms_pickle_path,
        pmi_pickle_path=args.pmi_pickle_path,
        output_path=args.output_file_path,
        e_th=args.e_th,
        c_th=args.c_th,
        m=args.m,
        similarity_factor=args.similarity_factor,
        probability_factor=args.probability_factor,
        length_factor=args.length_factor,
        bias_factor=args.bias_factor,
        number_of_extractions=args.number_of_extractions
    )
    contradiction_extractor.save_contradictory_pairs()

