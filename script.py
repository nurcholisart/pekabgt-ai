from langchain.prompts.prompt import PromptTemplate
from langchain.chains.conversational_retrieval.prompts import (
    CONDENSE_QUESTION_PROMPT,
)
from langchain.chains import ConversationalRetrievalChain
import tempfile
import requests
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory, RedisChatMessageHistory

import os

import logging

logging.getLogger("openai").setLevel(logging.DEBUG)  # logging.INFO or logging.DEBUG


redis_chat_history = RedisChatMessageHistory(session_id="chat_history", ttl=3600)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, chat_memory=redis_chat_history)

template = """You are an AI customer service agent for answering question about Qiscus. Your name is Peka.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
If user ask something not about Qiscus, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question", "context"])

vectorstore = FAISS.load_local("db", OpenAIEmbeddings())

chain = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(temperature=0, max_tokens=150),
    retriever=vectorstore.as_retriever(),
    return_source_documents=False,
    qa_prompt=prompt
)

question = "No. Please end the chat"

vectordbkwargs = {"search_distance": 0.9}

chat_history = []

result = chain(
    {
        "question": question,
        "chat_history": chat_history,
        "vectordbkwargs": vectordbkwargs
    }
)

# redis_chat_history.add_user_message(result["question"])
# redis_chat_history.add_ai_message(result["answer"])

print(result)
