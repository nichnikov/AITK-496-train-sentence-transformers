import os
import torch
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', )

logger = logging.getLogger()
logger.setLevel(logging.INFO)


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


def texts2vectors(texts: list[str], model: SentenceTransformer) -> torch.Tensor:
    """
    Переводит список текстов в эмбеддинги посредством модели SentenceTransformer

    Args:
        texts (list[str]): _description_
        model (SentenceTransformer): _description_

    Returns:
        _type_: _description_
    """
    txts_chunks = chunks(texts, 10000)
    tensors = []
    for txs in txts_chunks:
        embeddings = model.encode(txs, convert_to_tensor=True)
        tensors.append(embeddings)
    return torch.cat(tensors)


def cos_sim(tensors1: torch.Tensor, tensors2: torch.Tensor) -> list[torch.Tensor]:
    """Возвращает косинусную близость между тензорами из матрицы тензоров

    Args:
        tensors1 (torch.Tensor): _description_
        tensors2 (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """
    tensors_chunks1 = torch.split(tensors1, 10000)
    tensors_chunks2 = torch.split(tensors2, 10000)
    scores = []
    for tnsrs1, tnsrs2 in zip(tensors_chunks1, tensors_chunks2):
        cos_scrs = util.cos_sim(tnsrs1, tnsrs2)
        scores += [cos_scrs[i][i] for i in range(cos_scrs.shape[0])]
    return scores


def texts_similarity_estimation(sentences1: list[str], sentences2: list[str], model: SentenceTransformer) -> list[float]:
    """
    Возвращает для каждой текстовой пары из zip(sentences1, sentences2) скор похожести

    Args:
        sentences1 (list[str]): _description_
        sentences2 (list[str]): _description_
    """
    embeddings1 = texts2vectors(sentences1, model)
    embeddings2 = texts2vectors(sentences2, model)

    return [t.item() for t in cos_sim(embeddings1, embeddings2)]


if __name__ == "__main__":
    test_df = pd.read_feather(os.path.join("data", "sys_1_test_queries.feather"))

    test_quntity = 100000
    sentences1 = test_df["LmQuery1"].to_list()[:test_quntity]
    sentences2 = test_df["LmQuery2"].to_list()[:test_quntity]
    true_lbs = test_df["label"].to_list()[:test_quntity]

    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    # Compute embedding for both lists
    cosine_scores = texts_similarity_estimation(sentences1, sentences2, model)

    # Output the pairs with their score
    test_result = [{"query1": q1, 
                    "query2": q2,
                    "score": cs,
                    "true_lb": tl,
                    "predict_lb": score2lb(cs, 0.5)} for q1, q2, cs, tl in zip(sentences1, sentences2, cosine_scores, true_lbs)]


    test_result_df = pd.DataFrame(test_result)
    print(test_result_df)

    test_result_df.to_csv(os.path.join("results", "sys_1_test_results.csv"), sep="\t", index=False)