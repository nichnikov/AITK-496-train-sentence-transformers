from typing import Any
from collections import namedtuple
from itertools import chain
from src.texts_processing import TextsTokenizer
from sentence_transformers import SentenceTransformer, util


class QueriesAnalysis:
    def __init__(self, tokenizer: TextsTokenizer, model: SentenceTransformer) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.QueriesTuple = namedtuple("Queris", "Cluster, LemCluster")
        # self.QueriesTuple = namedtuple("Queris", "query, lem_query")

    def dissimilar_queries(self, max_examples: int, threshold: float, queries: [str]) -> []:
        """
        Функция, отбирающая для каждого быстрого ответа только непохожие вопросы (с точки зрения Сберт-трансформера)
        max_examples - столько примеров будет отбираться из быстрого ответа
        threshold - векторы примеров должны быть не ближе друг ко другу, чем это косинусное расстояние
        fa_ids - уникальные айди быстрых ответов
        """
        lm_queries = [" ".join(lq) for lq in self.tokenizer(queries)]
        paphrs = util.paraphrase_mining(self.model, lm_queries)
        dissim_queries = []
        if paphrs:
            if paphrs[-1][0] >= 0.5: 
                # в группе вопросы должны быть относительно похожими (больше, чем на 0.5 для Сберта), 
                # иначе группа совсем разнородная и лучше ее не использовать для обучения нейронной сети
                not_sims_paphrs = [(p[1], p[2]) for p in paphrs[-max_examples:] if p[0] <= threshold]
                if not_sims_paphrs:
                    paphrs_not_sim_indx_1, paphrs_not_sim_indx_2 = zip(*not_sims_paphrs)
                    lm_queries_temp_1 = [lm_queries[i] for i in paphrs_not_sim_indx_1]
                    lm_queries_temp_2 = [lm_queries[i] for i in paphrs_not_sim_indx_2]
                    temp_paphrs_1 = util.paraphrase_mining(self.model, lm_queries_temp_1)
                    temp_paphrs_2 = util.paraphrase_mining(self.model, lm_queries_temp_2)                                 
                    temp_paphrs_1_indx = [p for p in temp_paphrs_1 if p[0] <= threshold]                 
                    temp_paphrs_2_indx = [p for p in temp_paphrs_2 if p[0] <= threshold]
                    qrs_indx1 = [(paphrs_not_sim_indx_1[p[1]], paphrs_not_sim_indx_1[p[2]]) for p in temp_paphrs_1_indx[-max_examples:]]
                    qrs_indx2 = [(paphrs_not_sim_indx_2[p[1]], paphrs_not_sim_indx_2[p[2]]) for p in temp_paphrs_2_indx[-max_examples:]]
                    unique_indexes = list(set([x for x in chain(*(qrs_indx1 + qrs_indx2))]))
                    if unique_indexes:
                        finished_lm_queries = [lm_queries[i] for i in unique_indexes]
                        finished_paphrs = util.paraphrase_mining(self.model, finished_lm_queries)
                        sifted_indexes = [unique_indexes[p[1]] for p in finished_paphrs if p[0] > threshold]
                        for idx in [i for i in unique_indexes if i not in sifted_indexes]:
                            dissim_queries.append(self.QueriesTuple(queries[idx], lm_queries[idx]))
                    else:
                        dissim_queries.append(self.QueriesTuple(queries[0], lm_queries[0]))
                else:
                    dissim_queries.append(self.QueriesTuple(queries[0], lm_queries[0]))
        return dissim_queries

    def __call__(self, max_examples: int, threshold: float, quries: [str]) -> Any:
        return self.dissimilar_queries(max_examples, threshold, quries)