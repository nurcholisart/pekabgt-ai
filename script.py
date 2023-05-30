import os
import logging

from langchain.chains.llm import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.conversational_retrieval.prompts import (
    CONDENSE_QUESTION_PROMPT,
    QA_PROMPT,
)
from langchain.chains.question_answering import load_qa_chain

from langchain.prompts.prompt import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory, RedisChatMessageHistory

logging.getLogger("openai").setLevel(logging.DEBUG)  # logging.INFO or logging.DEBUG

llm = OpenAI(temperature=0, verbose=True)

question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=True)

template = """You are an AI customer service agent for answering question about Qiscus. Your name is Peka.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
If user ask something not about Qiscus, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question", "context"])

doc_chain = load_qa_chain(
    llm=llm, chain_type="stuff", prompt=prompt, verbose=True
)

vectorstore = FAISS.load_local("db", OpenAIEmbeddings())

redis_chat_history = RedisChatMessageHistory(session_id="chat_history:room_1212182", ttl=3600)

qa = ConversationalRetrievalChain(
    retriever=vectorstore.as_retriever(),
    combine_docs_chain=doc_chain,
    question_generator=question_generator,
    verbose=True,
)

chat_history = []

query = "Apa itu Robolabs?"
result = qa({"question": query, "chat_history": redis_chat_history.messages})

redis_chat_history.add_user_message(query)
redis_chat_history.add_ai_message(result["answer"])

print(result)