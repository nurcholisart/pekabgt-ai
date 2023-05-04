import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import (
    CONDENSE_QUESTION_PROMPT,
)

from langchain.prompts.prompt import PromptTemplate

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from bs4 import BeautifulSoup

import requests

os.environ["OPENAI_API_KEY"] = "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"


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

# ===========================================

# system_template = """Use the following pieces of context to answer the users question.
# Take note of the sources and include them in the answer in the list format.
# If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
# If the answer is not available in the context, just say "I don't know", don't try to make up an answer.
# If user no longer needs other information or wants to end the conversation, write "#close_chat" at the end of your answer.
# If the user want to communicate with human agent, write "#assign_agent" word at the end of your answer.
# ----------------
# {context}"""

template="""You are an AI customer service agent for answering question about Qiscus.
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

prompt = PromptTemplate(
    template=template, input_variables=["question", "context"]
)

chain = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(temperature=0),
    retriever=vectorstore.as_retriever(),
    condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    qa_prompt=prompt,
    return_source_documents=True,
)

result = chain({"question": "Lebih baik kaki kanan atau kiri?", "chat_history": []})
print(result)
