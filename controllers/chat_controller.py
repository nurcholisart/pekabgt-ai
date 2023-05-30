import tempfile
import requests

from langchain.prompts.prompt import PromptTemplate
# from langchain.chains.conversational_retrieval.prompts import (
#     CONDENSE_QUESTION_PROMPT,
# )
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI


class ChatController:
    def call(
        self,
        api_key: str,
        faiss_url: str,
        pkl_url: str,
        question: str,
        chat_history: list[tuple] = [],
        chatbot_name: str = "Peka",
        chatbot_description: str = "You are an AI customer service agent for answering question about Qiscus",
    ) -> dict:

        base_template = f"""{chatbot_description}. Your name is {chatbot_name}.
        Use the following pieces of context to answer the question at the end using Indonesia language.
        If you don't know the answer, just say that you don't know politely, don't try to make up an answer.
        If user ask something not about {chatbot_name}, just say that you don't know politely, don't try to make up an answer.
        
        """

        context = """
        {context}

        Question: {question}
        Helpful Answer:"""

        template = base_template + "\n" + context

        prompt = PromptTemplate(
            template=template, input_variables=["question", "context"]
        )

        result = {}

        with tempfile.TemporaryDirectory() as tempdirname:
            resp1 = requests.get(faiss_url, allow_redirects=True)

            with open("index.faiss", "wb") as file:
                file.write(resp1.content)

            resp2 = requests.get(pkl_url, allow_redirects=True)
            with open("index.pkl", "wb") as file:
                file.write(resp2.content)

            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            vectorstore = FAISS.load_local(".", embeddings)

            chain = ConversationalRetrievalChain.from_llm(
                llm=OpenAI(openai_api_key=api_key, temperature=0),
                retriever=vectorstore.as_retriever(),
                qa_prompt=prompt,
                return_source_documents=False,
            )

            vectordbkwargs = {"search_distance": 0.9}

            print("SAMPE SINIIIIIIIIIIIIIIIIIIIIIIIII")

            raw_result = chain(
                {
                    "question": question,
                    "vectordbkwargs": vectordbkwargs,
                    "chat_history": chat_history,
                }
            )

            result = {
                "question": raw_result["question"],
                "answer": raw_result["answer"],
                "chat_history": raw_result["chat_history"],
                "source_documents": [],
            }

            return result
