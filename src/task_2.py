import torch
from config import *
import torch.nn as nn
from task_1 import get_sentence_transformer


class ClassificationHead(nn.Module):
    """
    A nn.Module representing a single classification head of the multi-task
    sentence transformer.

    The head consists of a single fully connected layer, outputting the specific
    number of classes required by the task. This is the simplest architecture
    possible, and additions may be made to better suit specific tasks. 
    """
    def __init__(self, n):
        super(ClassificationHead, self).__init__()
        self.linear = nn.Linear(384, n)

    def forward(self, x):
        out = self.linear(x)
        return out


class SentenceTransformerWithHeads(nn.Module):
    """
    A nn.Module representing a sentence transformer with two classification heads
    attached to it. 
    
    The output dimension of these heads are specified with
    ``num_classes_a`` and ``num_classes_b``. During the forward pass, which head
    to use is specified by setting the ``head`` argument to 'A' or 'B'. 
    """
    def __init__(self, num_classes_a, num_classes_b):
        super(SentenceTransformerWithHeads, self).__init__()
        self.st = get_sentence_transformer()
        self.num_classes_a = num_classes_a
        self.num_classes_b = num_classes_b
        self.head_a = ClassificationHead(n=num_classes_a)
        self.head_b = ClassificationHead(n=num_classes_b)

    def forward(self, x, head):
        features = torch.tensor(self.st.encode(x)).to(device)
        if head == 'A':
            out = self.head_a(features)
        elif head == 'B':
            out = self.head_b(features)
        else:
            raise ValueError("Please specify 'A' or 'B' for head")
        return out