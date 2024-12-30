from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from Document_Ingestion.document_ingestion_neo4j import Neo4JDocumentManager
# from Document_Ingestion.documet_igestion_chroma import ChromaDocumentManager
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.callbacks.manager import get_openai_callback
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import ClassVar, List
from langchain_core.documents import Document
from langchain_chroma import Chroma


class CombinedVectorRetriever(BaseRetriever):
    neo4j_manager: ClassVar[Neo4JDocumentManager] = Neo4JDocumentManager()

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        neo4j_retriever = self.neo4j_manager.get_retriever()
        chroma_retriever = Chroma(
            collection_name="document-normal-rag",
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-large", show_progress_bar=True, dimensions=1536),
            persist_directory="/home/abhishek/Desktop/hybrid_rag/Document_Ingestion/chroma_instance"
        ).as_retriever()
        
        document_list = []
        try:
            chroma_documents = chroma_retriever.get_relevant_documents(query)
            neo4j_documents = neo4j_retriever.get_relevant_documents(query)
            document_list.extend(chroma_documents)
            document_list.extend(neo4j_documents)
            return document_list
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return []
        
        
        

class RagHandler:
   def __init__(self):
      self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
      self.embedding = OpenAIEmbeddings(model="text-embedding-3-large", show_progress_bar=True, dimensions=1536)
      self.retriever =  CombinedVectorRetriever()
      self.system_prompt = """
      You are a context-aware QA system analyzing information from multiple sources. 
      answer the user following question in professional form
      """

   def answer_question(self, question):
      prompt = ChatPromptTemplate.from_messages([
         SystemMessage(content=self.system_prompt),
         ("human", "Use the following context to answer the question:\n\nContext: {context}\n\nQuestion: {input}"),
         ("assistant", "I'll help answer your question based on the context provided.")
      ])
      qa_chain = create_retrieval_chain(
            retriever = self.retriever,
            combine_docs_chain=create_stuff_documents_chain(
               llm=self.llm,
               prompt=prompt
            )
      )

      with get_openai_callback() as cb:
            result = qa_chain.invoke({"input": question})
      return result.get("answer", "")


