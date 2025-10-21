# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
# ---


# %% [markdown]
# # 1.LangChainを用いたエージェントの作成

# 本レポートでは、Geminiに加えてLangChainを用いることで、与えられた背景を元にして回答するBotの作成を行なう。今回は、授業資料、論文などを元にして回答を生成するBotに焦点を当てる。

# # 2.方法

# まず、入力されたpdfからLangChainに組み込まれているpyPDF2およびsentence-transformerを用いてデータとしてGeminiが読み込める形にする。

import os
from typing import List

from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# embeddingモデルとしてSentenceTransformerを用いる
from sentence_transformers import SentenceTransformer

os.environ["GOOGLE_API_KEY"] = input("Enter your Gemini API key: ")

pdf = "./THE LOTTERY TICKET HYPOTHESIS.pdf"
loader = PyPDFLoader(pdf)
docs = loader.load()

# separate by chanks with 200 letters overlap
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

texts = splitter.split_documents(docs)
texts_list = [t.page_content for t in texts]

# light weight model
hf_model = SentenceTransformer("all-MiniLM-L6-V2")


class MyHFEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    # 文書リストをベクトルに変換
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [vec.tolist() for vec in self.model.encode(texts)]

    # 単一のクエリをベクトルに変換
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()


hf_embeddings = MyHFEmbeddings("all-MiniLM-L6-V2")
vectorstore = FAISS.from_texts(texts_list, embedding=hf_embeddings)
#
# # llm
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
qa = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, return_source_documents=True
)

# chatbot
try:
    while True:
        query = input("\nYou: ").strip()
        if not query:
            continue
        answer = qa.invoke({"query": query})
        print(f"\nGemini: {answer['result']}")
        print("\n=== Source Documents (excerpt) ===")
        for i, doc in enumerate(answer["source_documents"], start=1):
            excerpt = doc.page_content[:200].replace("\n", " ")
            print(f"{i}. {excerpt}…")

except KeyboardInterrupt:
    print("\n Exit")
