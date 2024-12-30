from typing import List, Optional
from langchain_chroma import Chroma
from langchain.schema import Document
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()


class ChromaDocumentManager:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-large", show_progress_bar=True, dimensions=1536)
        self.vector_db = Chroma(
            collection_name="document-normal-rag",
            embedding_function=self.embedding,
            persist_directory="./chroma_instance"
        )

    def get_retriver(self):
        chroma_retriver = Chroma(
            collection_name="document-normal-rag",
            embedding_function=self.embedding,
            persist_directory="./chroma_instance"
        )
        return chroma_retriver.as_retriever()

    def __ingest_document(self, document_source: str) -> Optional[List[Document]]:
        try:
            file_path = Path(document_source)
            
            if not file_path.exists():
                print(f"File not found: {document_source}")
                return None
                
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,  # Increased chunk size for better context
                    chunk_overlap=200,  # Increased overlap
                    length_function=len,
                )
                split_docs = text_splitter.split_documents(documents)
                
                if not split_docs:
                    print("No content extracted from PDF")
                    return None
                    
                return split_docs
            else:
                print(f"Unsupported file type: {file_path.suffix}")
                return None
                
        except Exception as e:
            print(f"Error during document ingestion: {str(e)}")
            return None

    def __ingest_vector_db(self, document_list: List[Document]) -> bool:
        if not document_list:
            print("Empty document list provided")
            return False
            
        try:
            self.vector_db.add_documents(documents=document_list)
            return True
        except Exception as e:
            print(f"Error during vector DB ingestion: {str(e)}")
            return False

    def ingest_document(self, document_source: str) -> bool:
        try:
            document_list = self.__ingest_document(document_source=document_source)
            if not document_list:
                return False
                
            return self.__ingest_vector_db(document_list=document_list)
        except Exception as e:
            print(f"Error during document processing: {str(e)}")
            return False