import torch
from tqdm import tqdm
import torch.nn as nn
import logging


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate(model, dataloader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids_a = batch['input_ids_a'].to(device)
            attention_mask_a = batch['attention_mask_a'].to(device)
            input_ids_b = batch['input_ids_b'].to(device)
            attention_mask_b = batch['attention_mask_b'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Accuracy: {acc:.4f}")
    logger.info(f"Accuracy {acc:.4f}")
    return acc


def train(model, train_loader, test_loader, optimizer, device, epochs=3):
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch + 1}")
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            input_ids_a = batch['input_ids_a'].to(device)
            attention_mask_a = batch['attention_mask_a'].to(device)
            input_ids_b = batch['input_ids_b'].to(device)
            attention_mask_b = batch['attention_mask_b'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        print("Test Accuracy")
        evaluate(model, test_loader, device)
    return model