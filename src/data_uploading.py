import re
from datetime import datetime


class DataUploading:
        def __init__(self, db_con) -> None:
            self.db_con = db_con
            self.data_dicts = []

        def sys_data_upload(self, sys_id) -> None:
            """получение данных из БД"""
            today = datetime.today().strftime('%Y-%m-%d')
            rows = self.db_con.get_rows(int(sys_id), today)
            self.data_dicts = [nt._asdict() for nt in rows]

        def get_clusters(self) -> [{}]:
            # Извлечение вопросов (кластеров)
            clusters = [{"id": d["ID"], "query": d["Cluster"]} for d in self.data_dicts]
            return clusters

        def get_answers(self) -> [{}]:
            # Извлечение текстов ответов
            patterns = re.compile("&#|;|\xa0")
            answers = [{"ID": patterns.sub(" ", str(tpl[0])), "DocName": patterns.sub(" ", str(tpl[1])), "ShortAnswer": tpl[2]} 
                    for tpl in set([(d["ID"], d["DocName"], d["ShortAnswerText"]) 
                                    for d in self.data_dicts])]
            return answers
