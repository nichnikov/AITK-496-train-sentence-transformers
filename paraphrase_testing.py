import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def score2lb(score: float, threshold: float) -> int:
    # assert score <= 1.0 and score >= 0.0
    assert threshold <= 1.0 and threshold >= 0.0
    
    if score > threshold:
        return 1
    else:
        return 0

def texts2vectors(texts: list[str], model: SentenceTransformer):
    txts_chunks = chunks(texts, 500)
    tensors = []
    for txs in  txts_chunks:
        embeddings = model.encode(txs, convert_to_tensor=True)
        tensors.append(embeddings)
    return torch.cat(tensors)

def cos_sim(tensors1: torch.Tensor, tensors2: torch.Tensor):
    tensors_chunks1 = torch.split(tensors1, 500)
    tensors_chunks2 = torch.split(tensors2, 500)
    scores = []
    for tnsrs1, tnsrs2 in zip(tensors_chunks1, tensors_chunks2):
        cos_scrs = util.cos_sim(tnsrs1, tnsrs2)
        scores.append(cos_scrs)
    return torch.cat(scores)


test_df = pd.read_feather(os.path.join("data", "sys_1_test_queries.feather"))
print(test_df)


test_quntity = 1000

sentences1 = test_df["LmQuery1"].to_list()[:test_quntity]
sentences2 = test_df["LmQuery2"].to_list()[:test_quntity]
true_lbs = test_df["label"].to_list()[:test_quntity]


model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')


# Compute embedding for both lists
embeddings1 = texts2vectors(sentences1, model)
embeddings2 = texts2vectors(sentences2, model)
cosine_scores = cos_sim(embeddings1, embeddings2)

# Output the pairs with their score
test_result = []
for i in range(len(sentences1[:test_quntity])):
    test_result.append({"query1": sentences1[i], 
                        "query2": sentences2[i],
                        "score": cosine_scores[i][i],
                        "true_lb": true_lbs[i],
                        "predict_lb": score2lb(cosine_scores[i][i], 0.5)
                        })

test_result_df = pd.DataFrame(test_result)
print(test_result_df)

test_result_df.to_csv(os.path.join("results", "sys_1_test_results.csv"), sep="\t", index=False)