import os

from langchain.chat_models import ChatOpenAI
# from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
# from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.schema import Document

from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import RedisChatMessageHistory
from langchain.callbacks import get_openai_callback

from fastapi.responses import JSONResponse

import requests

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class V2ChatController:
    def __init__(
        self,
        api_key: str,
        session_id: str,
        question: str,
        system_prompt: str,
        faiss_url: str,
        pkl_url: str,
    ) -> None:
        self.api_key = api_key
        self.session_id = session_id
        self.question = question
        self.system_prompt = system_prompt
        self.faiss_url = faiss_url
        self.pkl_url = pkl_url

        self.condense_prompt_template = self.__set_condense_prompt_template()
        self.user_prompt_template = self.__set_user_prompt_template()
        self.message_history_store = self.__set_message_history_store()
        self.chat_llm = self.__set_chat_llm()
        self.embedding_wrapper = self.__set_embedding_wrapper()
        self.vectorstore = self.__get_vectorstore()

    def call(self) -> JSONResponse:
        with get_openai_callback() as cb:
            try:
                retriever = self.vectorstore.as_retriever()
                retriever.search_kwargs = {"k": 3, "search_distance": 0.9}

                qa = ConversationalRetrievalChain.from_llm(
                    llm=self.chat_llm,
                    retriever=retriever,
                    condense_question_prompt=self.condense_prompt_template,
                    condense_question_llm=self.chat_llm,
                    combine_docs_chain_kwargs={"prompt": self.user_prompt_template},
                    chain_type="stuff",
                    return_source_documents=True,
                )

                result = qa(
                    {
                        "question": self.question,
                        "chat_history": self.message_history_store.messages,
                    }
                )

                self.message_history_store.add_user_message(result["question"])
                self.message_history_store.add_ai_message(result["answer"])

                source_documents: list[Document] = result["source_documents"]

                source_documents_dict = []
                for sd in source_documents:
                    source_documents_dict.append(sd.dict())

                dict_response = {
                    "success": True,
                    "question": result["question"],
                    "answer": result["answer"],
                    "chat_history": [],
                    "source_documents": source_documents_dict,
                    "meta": {
                        "prompt_tokens": cb.prompt_tokens,
                        "completion_tokens": cb.completion_tokens,
                        "total_tokens": cb.total_tokens,
                        "total_cost": cb.total_cost,
                        "success_requests": cb.successful_requests,
                    },
                }

                return JSONResponse(dict_response)
            except Exception as e:
                return JSONResponse(
                    {
                        "success": False,
                        "message": f"Request to AI service failed. {str(e)}",
                        "meta": {
                            "prompt_tokens": cb.prompt_tokens,
                            "completion_tokens": cb.completion_tokens,
                            "total_tokens": cb.total_tokens,
                            "total_cost": cb.total_cost,
                            "success_requests": cb.successful_requests,
                        },
                    },
                    status_code=500,
                )

    def __set_message_history_store(self) -> RedisChatMessageHistory:
        FIFTEEN_MINUTES = 900

        return RedisChatMessageHistory(
            session_id=f"chat_history:{self.session_id}",
            ttl=FIFTEEN_MINUTES,
            url=REDIS_URL,
        )

    def __set_user_prompt_template(self) -> str:
        _template = """CONTEXT:
        {context}

        QUESTION: {question}
        HELPFUL ANSWER:"""

        return PromptTemplate(
            template=("\n" + self.system_prompt + "\n\n" + _template),
            input_variables=["question", "context"],
        )

    def __set_condense_prompt_template(self) -> str:
        _template = """Given the following conversation and a follow up input, rephrase the follow up input to be a standalone input, in its original language.

        Chat History:
        {chat_history}

        Follow Up Input:
        {question}

        Standalone input:
        """

        return PromptTemplate.from_template(_template)

    def __set_chat_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            openai_api_key=self.api_key, temperature=0, model="gpt-3.5-turbo-16k"
        )

    def __set_embedding_wrapper(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(openai_api_key=self.api_key)

    def __get_vectorstore(self) -> FAISS:
        faiss_file_name = self.faiss_url.split("/")[-1]
        directory_name = faiss_file_name.split(".faiss")[0]
        tmp_directory_name = f"./tmp/{directory_name}"

        if not os.path.exists(tmp_directory_name):
            self.___download_faiss_and_pkl_url()

        vectorstore = FAISS.load_local(
            tmp_directory_name, embeddings=self.embedding_wrapper
        )

        return vectorstore

    def ___download_faiss_and_pkl_url(self) -> None:
        get_faiss_resp = requests.get(self.faiss_url, allow_redirects=True)

        faiss_directory_name = self.faiss_url.split("/")[-1].split(".faiss")[0]
        faiss_tmp_directory_name = f"./tmp/{faiss_directory_name}"
        faiss_file_name = f"{faiss_tmp_directory_name}/index.faiss"

        os.makedirs(faiss_tmp_directory_name, exist_ok=True)
        with open(faiss_file_name, "xb") as file:
            file.write(get_faiss_resp.content)

        get_pkl_resp = requests.get(self.pkl_url, allow_redirects=True)

        pkl_directory_name = self.pkl_url.split("/")[-1].split(".pkl")[0]
        pkl_tmp_directory_name = f"./tmp/{pkl_directory_name}"
        pkl_file_name = f"{pkl_tmp_directory_name}/index.pkl"

        os.makedirs(pkl_tmp_directory_name, exist_ok=True)
        with open(pkl_file_name, "xb") as file:
            file.write(get_pkl_resp.content)
