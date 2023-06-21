from models import Article

from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter, MarkdownTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from utils import qiscus

import time


class EmbeddingController:
    def call(self, api_key: str, app_code: str, articles: list[Article]) -> dict:
        documents = []

        for article in articles:
            document = self._generate_langchain_document(article)
            documents.append(document)

        documents = self._split_character_text(documents)
        result = self._generate_and_save_vectorstore(api_key, app_code, documents)

        # upload faiss
        faiss_url = self._upload_file_to_qiscus(app_code, result["faiss_path"])

        # upload pkl
        pkl_url = self._upload_file_to_qiscus(app_code, result["pkl_path"])

        return {"faiss_url": faiss_url, "pkl_url": pkl_url}

    def _generate_langchain_document(self, article: Article) -> Document:
        document = Document(
            page_content=article.content,
            metadata={"title": article.title, "link": article.link},
        )

        return document

    def _convert_html_to_plain(self, content: str) -> str:
        soup = BeautifulSoup(content, "html.parser")
        plain_content = soup.get_text()

        return plain_content

    def _split_character_text(self, documents: list[Document]) -> list[Document]:
        text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=0)

        # text_splitter = CharacterTextSplitter(
        #     chunk_size=500,
        #     chunk_overlap=0,
        #     separator="\n##"
        # )

        splitted_documents = text_splitter.split_documents(documents)
        return splitted_documents

    def _generate_and_save_vectorstore(
        self, api_key: str, app_code: str, documents: list[Document]
    ) -> FAISS:
        current_time = str(int(time.time() * 1000))
        index_name = f"{app_code}_embeddings_{current_time}"

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local("tmp", index_name)

        faiss_path = f"tmp/{index_name}.faiss"
        pkl_path = f"tmp/{index_name}.pkl"

        return {"faiss_path": faiss_path, "pkl_path": pkl_path}

    def _upload_file_to_qiscus(self, app_code: str, file_path: str) -> str:
        document_url = qiscus.upload_file(app_code, file_path)
        return document_url
