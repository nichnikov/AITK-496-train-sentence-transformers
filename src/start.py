import os
import pandas as pd
from src.data_uploading import DataUploading
# from src.storage import DataFromDB
# from src.queries_analysis import QueriesAnalysis
from src.texts_processing import TextsTokenizer
from sentence_transformers import SentenceTransformer

db_credentials =  {
        "server_host": "statistics.sps.hq.amedia.tech",
        "user_name": "nichnikov_ro",
        "password": "220929SrGHJ#yu"}

# b_con = DataFromDB(**db_credentials)
# data_upload = DataUploading(db_con)

tokenizer = TextsTokenizer()

stopwords = []
for fn in ["greetings.csv", "stopwords.csv"]:
    stwrs_df = pd.read_csv(os.path.join(os.getcwd(), "data", fn), sep="\t")
    stopwords += list(stwrs_df["stopwords"])

tokenizer.add_stopwords(stopwords)
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# queries_analysis = QueriesAnalysis(tokenizer, model)