# CEMaT: Contradiction Extraction from Medical Texts using Natural Language Inference


This is the implementation of the paper CEMaT: Contradiction Extraction from Medical Texts using Natural Language Inference.

## Overview
![Architecture of CEMaT: 1- Siamese Network is trained on NLI datasets. 2- Medical Texts Filtered by name of Drug/Disease 3- Using trained siamese network retrieved texts covert to their embeddings 4- Select representative points](./cemat.png)

To train Siamese network use following command

```bash
python train_siamese_network.py