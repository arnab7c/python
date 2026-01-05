import asyncio
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_community.chat_message_histories import ChatMessageHistory
from threading import Event

from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

import src.Utility as Utility

logger = Utility.setup_logger("Retriever.log")

class Retriever:
    def __init__(self):
        logger.info("Starting Retriever module.")
        self.rag_chain = self.set_rag_chain()

    def __call__(self,stop_event:Event = None):

        self.stop_event = stop_event

        if self.stop_event.is_set():
            logger.info("Stopped before ingestion start")
            return
        logger.info("Starting Retriever module.")

# =====================================================
#
# =====================================================
    def set_llm(self):
        llm_provider = Utility.get_llm_provider()
        llm_model = Utility.get_llm_model(llm_provider)
        temperature = Utility.get_temperature(llm_provider)

        llm = ChatOllama(
            model=llm_model,
            temperature=temperature,
        )
        return llm

# =====================================================
#
# =====================================================
    def set_rag_chain(self):
        llm = self.set_llm()
        embedding = Utility.get_embedding_type()
        model_name = Utility.get_model_name(embedding)
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vector_db_type = Utility.get_vector_db()
        persist_dir = Utility.get_vector_dir(vector_db_type)

        # set up vector store
        vectorstore = Chroma(
            collection_name="GENAI",
            embedding_function=embeddings,
            persist_directory=persist_dir
        )

        # Create a prompt template
        prompt_template = """Use the following pieces of context to answer the human's question. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Human: {question}

        R2D2: """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )

        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={
                "prompt": prompt
            },
            return_source_documents=True,
        )

        return rag_chain

# =====================================================
# Test Code For Console Chat
# =====================================================
#     def chat_bot(self):
#         while True:
#             prompt = input("Enter your prompt: ")
#             if prompt.lower() == 'quit':
#                 print("Thank you for using the Chatbot. Goodbye!")
#                 break
#             response = self.rag_chain.invoke(prompt)
#             print(f"\nR2D2 => ",response["result"])

# =====================================================
# ASYNC ENTRY POINT (FastAPI SAFE)
# =====================================================
    async def run(self,prompt:str) -> dict:
        logger.info("Running RAG query", extra={"prompt": prompt})
        result = await asyncio.to_thread(
            self.rag_chain,
            {"query": prompt}
        )

        return {
            "answer": result.get("result", ""),
            "sources": [
                doc.metadata.get("source")
                for doc in result.get("source_documents", [])
            ],
        }
