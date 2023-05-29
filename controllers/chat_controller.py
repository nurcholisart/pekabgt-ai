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
        base_template = f"""{chatbot_description}. Your name is {chatbot_name}."""

        common_template = """You will answer the question using Indonesia language in conversational manner.
        You are given the following extracted parts of a long document and a question.
        Don't try to make up an answer. If you don't know the answer, just say "Maaf, saya tidak mengetahuinya" followed by "#dont_know" word.
        If the question is not about Qiscus, politely inform them that you are tuned to only answer questions about Qiscus. But, you must still answer the small talk.
        If the customer want to talk to a human agent, you have to ask them to wait the agent join the room followed by "#assign_agent" word.
        If the customer want to end the conversation, thank them politely followed by "#end_chat" word.
        Question: {question}
        =========
        {context}
        =========
        Answer:"""

        template = base_template + "\n" + common_template

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
                llm=OpenAI(openai_api_key=api_key),
                retriever=vectorstore.as_retriever(),
                condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                qa_prompt=prompt,
                return_source_documents=True,
            )

            raw_result = chain({"question": question, "chat_history": chat_history})

            documents = []
            source_documents = raw_result["source_documents"] or []
            for sd in source_documents:
                documents.append(
                    {"page_content": sd.page_content, "metadata": sd.metadata}
                )

            result = {
                "question": raw_result["question"],
                "answer": raw_result["answer"],
                "chat_history": raw_result["chat_history"],
                "source_documents": documents,
            }

            return result
