import logging
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import src.Utility as Utility
from threading import Event

class Ingestion:

    # def __init__(self, source_file:str, source_file_type:str):
    #     self.source_file = source_file
    #     self.source_file_type = source_file_type
    #     log_file = "Ingestion.log"
    #     Utility.set_logger(log_file)
    #     logging.info("Starting Ingestion module.")
    #     logging.info(f"Source file: {self.source_file}")
    #
    #     if source_file_type == "pdf":
    #         documentList = self.pdf_loader()
    #     else:
    #         raise ValueError("Unknown file source type")
    #     docChunks = self.splitter(documentList)
    #     self.vectorizer(docChunks)

    def __init__(self):
        pass

    def __call__(self, source_file:str, source_file_type:str, stop_event:Event = None):
        self.source_file = source_file
        self.source_file_type = source_file_type
        self.stop_event = stop_event
        self.logger = Utility.setup_logger("FileLoader.log")
        if self.stop_event.is_set():
            self.logger.info("Stopped before ingestion start")
            return
        self.logger.info("Starting Ingestion module.")
        self.logger.info(f"Source File: {self.source_file} and Source File Type: {self.source_file_type}")

        if source_file_type == "pdf":
            documentList = self.pdf_loader()
        else:
            self.logger.error("Unknown file source type")
            return
        docChunks = self.splitter(documentList)
        self.vectorizer(docChunks)

# =====================================================
# PDF loader function
# =====================================================
    def pdf_loader(self) -> list[Document] | None:
        if self.stop_event.is_set():
            self.logger.info("Stopped before pdf loader start")
            return
        data_dir = Utility.get_raw_data_directory()
        base_dir = Utility.get_base_dir()
        loader = PyPDFLoader(os.path.join(base_dir, data_dir, self.source_file))
        documentList = loader.load()
        self.logger.info(f"Document Loaded. Total Pages: {len(documentList)}")
        return documentList

# =====================================================
# Split document in to chunks based on chunk size defined in base.yaml
# =====================================================
    def splitter(self, documentList:list) -> list[Document] | None:
        if self.stop_event.is_set():
            self.logger.info("Stopped before splitter start")
            return
        splitter = Utility.get_splitter()
        chunk_size = Utility.get_chunk_size(splitter)
        chunk_overlap = Utility.get_chunk_overlap(splitter)
        # if new splitter added in base.yaml then add in below condition
        if splitter == "RecursiveCharacterTextSplitter":
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            self.logger.error("Unknown splitter type")
            return
        docChunks = text_splitter.split_documents(documentList)
        self.logger.info(f"Document Split into {len(docChunks)} smaller chunks.")
        return docChunks

# =====================================================
# Load chunks in vector db
# =====================================================
    def vectorizer(self, docChunks:list):
        if self.stop_event.is_set():
            self.logger.info("Stopped before vector load start")
            return
        embedding = Utility.get_embedding_type()
        model_name = Utility.get_model_name(embedding)
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vector_db_type = Utility.get_vector_db()
        persist_dir = Utility.get_vector_dir(vector_db_type)
        if vector_db_type == "chroma":
            Chroma.from_documents(documents=docChunks, embedding=embeddings,
                                  collection_name="GENAI",persist_directory=persist_dir)
            self.logger.info("Knowledge stored in Chroma.")
        else:
            self.logger.error("Unknown vector db type")
            return
