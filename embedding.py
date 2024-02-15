import os
import pandas as pd
import json
import string
import random
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


model_name = "OrdalieTech/Solon-embeddings-large-0.1"

embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,  # We recommend starting with a small model
    model_kwargs={"device": "cuda"},  # Or cuda if you have GPU
    encode_kwargs={"show_progress_bar": True},  # Show the progress of embeddings
    multi_process=False,
)  # set to True if you have mutliprocessing


df_docs = pd.read_csv("raw_data/lemonde_bunka.csv")
df_docs = df_docs[:1000000]
sentences = list(df_docs["titles"])
ids = list(df_docs["0"])

characters = string.ascii_letters + string.digits
random_string = "".join(random.choice(characters) for _ in range(20))

df_loader = pd.DataFrame(sentences, columns=["text"])
df_loader["doc_id"] = ids

loader = DataFrameLoader(df_loader, page_content_column="text")
documents_langchain = loader.load()


# the actual steps to embeddings
vectorstore = Chroma.from_documents(
    documents_langchain, embedding_model, collection_name=random_string
)

bunka_ids = [item["doc_id"] for item in vectorstore.get()["metadatas"]]
bunka_docs = vectorstore.get()["documents"]


bunka_embeddings = vectorstore._collection.get(include=["embeddings"])["embeddings"]


# Create a dictionary where IDs are keys and embeddings are values
embedding_dump = dict(zip(ids, bunka_embeddings))

with open("exports/embedding_lemonde_million.json", "w") as json_file:
    json.dump(embedding_dump, json_file)
