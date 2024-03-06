import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader

model = SentenceTransformer("distilbert-base-nli-mean-tokens")
train_examples = [
    InputExample(texts=["My first sentence", "My second sentence"], label=0.8),
    InputExample(texts=["Another pair", "Unrelated sentence"], label=0.3),
]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)