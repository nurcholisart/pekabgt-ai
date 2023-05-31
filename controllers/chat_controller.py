import tempfile
import requests

from fastapi.responses import JSONResponse

from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.base_language import BaseLanguageModel

import logging

logging.getLogger("openai").setLevel(logging.DEBUG)  # logging.INFO or logging.DEBUG

condense_template = """Given the following chat history and a follow up question, rephrase the follow up input question to be a standalone question.
Or end the conversation if it seems like it's done.
Chat History:\"""
{chat_history}
\"""
Follow Up Input: \"""
{question}
\"""
Standalone question:"""

condense_prompt = PromptTemplate.from_template(condense_template)


class NoOpLLMChain(LLMChain):
    """No-op LLM chain."""

    def __init__(self, llm: BaseLanguageModel):
        """Initialize."""
        super().__init__(
            llm=llm, prompt=PromptTemplate(template="", input_variables=[])
        )

    async def arun(self, question: str, *args, **kwargs) -> str:
        return question


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
        with get_openai_callback() as cb:
            try:
                llm = OpenAI(openai_api_key=api_key, max_tokens=500, verbose=True)

                base_template = f"""{chatbot_description}. Your name is {chatbot_name}.
                Use the following pieces of context and chat history to answer the question at the end using Indonesia language.
                If you don't know the answer, just say that you don't know politely, don't try to make up an answer.
                If user ask something not about {chatbot_name}, just say that you don't know politely, don't try to make up an answer.                
                """

                context = """
                Chat History:
                {chat_history}
                
                Context:
                {context}

                Question:
                {question}
                
                Helpful Answer:"""

                template = base_template + "\n" + context

                prompt = PromptTemplate(
                    template=template,
                    input_variables=["chat_history", "context", "question"],
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

                    question_generator = NoOpLLMChain(llm=llm)
                    doc_chain = load_qa_chain(
                        llm=llm, chain_type="stuff", verbose=True, prompt=prompt
                    )

                    chain = ConversationalRetrievalChain(
                        retriever=vectorstore.as_retriever(),
                        question_generator=question_generator,
                        combine_docs_chain=doc_chain,
                        verbose=True,
                    )

                    # chain = ConversationalRetrievalChain.from_llm(
                    #     llm=OpenAI(
                    #         openai_api_key=api_key, temperature=0, max_tokens=150
                    #     ),
                    #     retriever=vectorstore.as_retriever(),
                    #     return_source_documents=False,
                    #     condense_question_prompt=condense_prompt,
                    #     combine_docs_chain_kwargs={"prompt": prompt},
                    # )

                    raw_result = chain(
                        {
                            "question": question,
                            "vectordbkwargs": {"search_distance": 0.9},
                            "chat_history": chat_history,
                            "meta": {},
                        }
                    )

                    result = {
                        "success": True,
                        "question": raw_result["question"],
                        "answer": raw_result["answer"],
                        "chat_history": raw_result["chat_history"],
                        "source_documents": [],
                        "meta": {
                            "prompt_tokens": cb.prompt_tokens,
                            "completion_tokens": cb.completion_tokens,
                            "total_tokens": cb.total_tokens,
                            "total_cost": cb.total_cost,
                            "success_requests": cb.successful_requests,
                        },
                    }

                    return JSONResponse(result, status_code=200)
            except Exception as e:
                print(e)
                result = {
                    "success": False,
                    "message": f"Request to AI service failed. {str(e)}",
                    "meta": {
                        "prompt_tokens": cb.prompt_tokens,
                        "completion_tokens": cb.completion_tokens,
                        "total_tokens": cb.total_tokens,
                        "total_cost": cb.total_cost,
                        "success_requests": cb.successful_requests,
                    },
                }

                return JSONResponse(result, status_code=500)
