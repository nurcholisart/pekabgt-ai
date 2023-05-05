import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.chains.conversational_retrieval.prompts import (
    CONDENSE_QUESTION_PROMPT,
)

from langchain.prompts.prompt import PromptTemplate

from bs4 import BeautifulSoup

import requests

os.environ["OPENAI_API_KEY"] = ""


def get_article():
    resp = requests.get("https://kenowlejbes.000webhostapp.com/wp-json/wp/v2/posts/27")
    json_resp = resp.json()

    return {
        "link": json_resp["link"],
        "content": json_resp["content"]["rendered"],
        "title": json_resp["title"]["rendered"],
    }


# article = get_article()

article = {
    "link": "https://kenowlejbes.000webhostapp.com/2023/04/troubleshooting-error-while-validating-token-jwt",
    "content": '\n<ol>\n<li>You must make sure your AppId and Secret Key SDK Qiscus are correct.</li>\n\n\n\n<li>Make sure that the JWT made by your server is valid, to check valid or not JWT can check on jwt.io (http://jwt.io/).</li>\n\n\n\n<li>The last step, you have to make sure the server to make JWT must have a valid GMT time.&nbsp;To make sure, please do Synchronize Time from the Network (NTP) on your server. If your server is Linux OS, you can use timeset to synchronize your time network. Run it from your terminal.</li>\n</ol>\n\n\n\n<p>You can read :&nbsp;<a href="https://documentation.qiscus.com/chat-sdk-android/authentications">https://documentation.qiscus.com/chat-sdk-android/authentications</a>&nbsp;for further reference on our Qiscus Chat SDK authentication mechanism.</p>\n',
    "title": "Troubleshooting Error While Validating Token (JWT)",
}

soup = BeautifulSoup(article["content"], "html.parser")

content = soup.get_text()

doc = Document(
    page_content=content,
    metadata={
        "title": article["title"],
        "link": article["link"],
        "source": article["link"],
    },
)

documents = [
    Document(
        page_content=content,
        metadata={
            "title": article["title"],
            "link": article["link"],
            "source": article["link"],
        },
    )
]

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local(".tmp")

# ===========================================

template = """You are an AI customer service agent for answering question about Qiscus.
You are given the following extracted parts of 
a long document and a question. Provide a conversational answer using Indonesian language.
If you don't know the answer, just say "Hmm, I'm not sure." accompanied with word "#dont_know".
Don't try to make up an answer. If the question is not about
Qiscus and not available in the document, politely inform them that you are tuned
to only answer questions about Qiscus.
Question: {question}
=========
{context}
=========
Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question", "context"])

chain = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(temperature=0),
    retriever=vectorstore.as_retriever(),
    condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    qa_prompt=prompt,
    return_source_documents=True,
)

result = chain({"question": "Lebih baik kaki kanan atau kiri?", "chat_history": []})

related_documents = []
source_documents = result["source_documents"]
for sd in source_documents:
    related_documents.append({
        "page_content": sd.page_content,
        "metadata": sd.metadata
    })

cleaned_result = {
    "question": result["question"],
    "answer": result["answer"],
    "chat_history": result["chat_history"],
    "source_documents": related_documents
}

print(cleaned_result)
