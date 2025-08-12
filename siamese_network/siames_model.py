import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class SiameseBertClassifier(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased', no_unfreeze_layer=4, mlp_number_of_neurons=[128, 64],
                 dropout_rate=0.2, num_labels=3):
        super(SiameseBertClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(pretrained_model)
        config = AutoConfig.from_pretrained(pretrained_model)

        # we just fine tune last 4 layers of BERT model to avoid overfitting
        for param in self.model.parameters():
            param.requires_grad = False
        for i in range(config.num_hidden_layers - no_unfreeze_layer, config.num_hidden_layers):
            for param in self.model.encoder.layer[i].parameters():
                param.requires_grad = True

        hidden_size = self.model.config.hidden_size
        combined_dim = hidden_size * 4  # u, v, |u-v|, u*v

        layers = []
        prev_dim = combined_dim
        for dim in mlp_number_of_neurons:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            if dropout_rate != 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_labels))
        self.classifier = nn.Sequential(*layers)

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        u = self.model(input_ids=input_ids_a, attention_mask=attention_mask_a).last_hidden_state[:, 0, :]
        v = self.model(input_ids=input_ids_b, attention_mask=attention_mask_b).last_hidden_state[:, 0, :]

        abs_diff = torch.abs(u - v)
        elem_mult = u * v
        combined = torch.cat([u, v, abs_diff, elem_mult], dim=1)

        return self.classifier(combined)
