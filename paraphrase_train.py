"""
https://www.sbert.net/examples/training/sts/README.html
"""

import os
import pandas as pd
from src.start import tokenizer
from sentence_transformers import (SentenceTransformer, 
                                   InputExample, 
                                   evaluation, 
                                   losses)
from torch.utils.data import DataLoader

"Query1", "Query2"

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
train_df = pd.read_feather(os.path.join("data", "sys_1_train_queries.feather"))
val_df = pd.read_feather(os.path.join("data", "sys_1_val_queries.feather"))
train_df_ = train_df[:1000]
val_df_ = val_df[:1000]


train_lm_queries1 = [" ".join(l_) for l_ in tokenizer(train_df_["Query1"].to_list())]
train_lm_queries2 = [" ".join(l_) for l_ in tokenizer(train_df_["Query2"].to_list())]
train_labels = train_df_["label"].to_list()

val_lm_queries1 = [" ".join(l_) for l_ in tokenizer(val_df_["Query1"].to_list())]
val_lm_queries2 = [" ".join(l_) for l_ in tokenizer(val_df_["Query2"].to_list())]
val_labels = train_df_["label"].to_list()

evaluator = evaluation.EmbeddingSimilarityEvaluator(val_lm_queries1, val_lm_queries2, [float(x) for x in val_labels])
train_examples = [InputExample(texts=[sen1, sen2], label=float(lb)) for sen1, sen2, lb in zip(train_lm_queries1, train_lm_queries2, train_labels)]

# train_dataset = SentencesDataset(train_examples, model)

"""https://www.sbert.net/docs/training/overview.html"""

# Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

model_save_path = os.path.join("models", "sys1.paraphrase")
model.fit(train_objectives=[(train_dataloader, train_loss)], 
          epochs=5, 
          warmup_steps=100,
          evaluator=evaluator,
          evaluation_steps=50, 
          output_path=model_save_path)

model.save(os.path.join("models", "sys1.paraphrase"))