import torch
import torch.nn as nn
from task_2 import SentenceTransformerWithHeads
from config import *
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from itertools import chain
from datasets import load_dataset
from torch.utils.data import DataLoader


def train(dataloader_a, dataloader_b, model, loss_fn, optimizer):
    num_batches = len(dataloader_a)
    model.train()
    # Dataloaders are iterated through at the same time
    for i, (batch_a, batch_b) in enumerate(zip(dataloader_a, dataloader_b)):
        X_a = batch_a['text']
        y_a = batch_a['label'].to(device)
        X_b = batch_b['text']
        y_b = batch_b['label'].to(device)
        
        # Compute prediction error
        pred_a = model(X_a, 'A')
        loss_a = loss_fn(pred_a, y_a)
        pred_b = model(X_b, 'B')
        loss_b = loss_fn(pred_b, y_b)
        # Losses are combined through addition
        loss = loss_a + loss_b 

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 5 == 4:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{i+1:>5d}/{num_batches:>5d}]")


def test(dataloader_a, dataloader_b, model, loss_fn):
    num_batches = len(dataloader_a)
    model.eval()
    test_loss = 0
    
    # Metrics tracked include accuracy, f1-score, precision, and recall
    accuracy = MulticlassAccuracy()
    f1_a = MulticlassF1Score(num_classes=model.num_classes_a, average=None)
    precision_a = MulticlassPrecision(num_classes=model.num_classes_a, average=None)
    recall_a = MulticlassRecall(num_classes=model.num_classes_a, average=None)

    f1_b = MulticlassF1Score(num_classes=model.num_classes_b, average=None)
    precision_b = MulticlassPrecision(num_classes=model.num_classes_b, average=None)
    recall_b = MulticlassRecall(num_classes=model.num_classes_b, average=None)
    
    with torch.no_grad():
        for batch_a, batch_b in zip(dataloader_a, dataloader_b):
            X_a = batch_a['text']
            y_a = batch_a['label'].to(device)
            pred_a = model(X_a, 'A')

            # Compute evaluation metrics for task A
            test_loss += loss_fn(pred_a, y_a).item()
            accuracy.update(pred_a, y_a)
            f1_a.update(pred_a, y_a)
            precision_a.update(pred_a, y_a)
            recall_a.update(pred_a, y_a)

            X_b = batch_b['text']
            y_b = batch_b['label'].to(device)
            pred_b = model(X_b, 'B')

            # Compute evaluation metrics for task B
            test_loss += loss_fn(pred_b, y_b).item()
            accuracy.update(pred_b, y_b)
            f1_b.update(pred_b, y_b)
            precision_b.update(pred_b, y_b)
            recall_b.update(pred_b, y_b)
            
    test_loss /= num_batches
    loss_str = f"Avg loss: {test_loss:>8f}"
    accuracy_str = f"Accuracy: {(100*accuracy.compute()):>0.1f}%"

    precision_str_a = f"Precision (A): {precision_a.compute()}"
    recall_str_a = f"Recall (A): {recall_a.compute()}"
    f1_str_a = f"F1-Score (A): {f1_a.compute()}"

    precision_str_b = f"Precision (B): {precision_b.compute()}"
    recall_str_b = f"Recall (B): {recall_b.compute()}"
    f1_str_b = f"F1-Score (B): {f1_b.compute()}"

    print(f"""Test Error:
    {loss_str}
    {accuracy_str}
    
    {precision_str_a}
    {recall_str_a}
    {f1_str_a}
    
    {precision_str_b}
    {recall_str_b}
    {f1_str_b}
    """)


if __name__ == "__main__":
    dataset_a = "fancyzhx/ag_news"
    train_dataset_a = load_dataset(dataset_a, split="train").with_format("torch")
    test_dataset_a = load_dataset(dataset_a, split="test").with_format("torch")

    batch_size_a = 2000
    train_dataloader_a = DataLoader(train_dataset_a, batch_size=batch_size_a)
    test_dataloader_a = DataLoader(test_dataset_a, batch_size=batch_size_a)

    dataset_b = "zeroshot/twitter-financial-news-sentiment"
    train_dataset_b = load_dataset(dataset_b, split="train").with_format("torch")
    test_dataset_b = load_dataset(dataset_b, split="validation").with_format("torch")

    # Batch sizes are different so that both dataloaders can be iterated through at the same time
    batch_size_b = 160 
    train_dataloader_b = DataLoader(train_dataset_b, batch_size=batch_size_b)
    test_dataloader_b = DataLoader(test_dataset_b, batch_size=batch_size_b)

    model = SentenceTransformerWithHeads(num_classes_a=4, num_classes_b=3).to(device)
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(chain(model.head_a.parameters(), model.head_b.parameters()))

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader_a, train_dataloader_b, model, loss_fn, optimizer)
        test(test_dataloader_a, test_dataloader_b, model, loss_fn)