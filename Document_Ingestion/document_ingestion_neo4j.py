from langchain_neo4j.graphs import neo4j_graph
from langchain_neo4j import Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

os.environ['OPENAI_API_KEY'] = "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"
 
class Neo4JDocumentManager:
    def __init__(self):
        self.graph = neo4j_graph.Neo4jGraph(
            url="neo4j+s://6f86dca7.databases.neo4j.io",
            username="neo4j",
            password="aIbsnoZLGdcbGcrdqdZMFlGIYje5y-3Df7rxhXvxpKs"
        )
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-large",show_progress_bar=True, dimensions=1536)
        self.doc_transformer = LLMGraphTransformer(llm = ChatOpenAI(model="gpt-3.5-turbo"))
 
    def get_retriever(self):
        chunk_vector = Neo4jVector.from_existing_index(
            self.embedding,
            graph=self.graph,
            index_name="chunkVector",
            embedding_node_property="textEmbedding",
            text_node_property="text",
            retrieval_query="""
            // get the document
            MATCH (node)-[:PART_OF]->(d:Document)
            WITH node, score, d

            // get the entities and relationships for the document
            MATCH (node)-[:HAS_ENTITY]->(e)
            MATCH p = (e)-[r]-(e2)
            WHERE (node)-[:HAS_ENTITY]->(e2)

            // unwind the path, create a string of the entities and relationships
            UNWIND relationships(p) as rels
            WITH
                node,
                score,
                d,
                collect(apoc.text.join(
                    [labels(startNode(rels))[0], startNode(rels).id, type(rels), labels(endNode(rels))[0], endNode(rels).id]
                    ," ")) as kg
            RETURN
                node.text as text, score,
                {
                    document: d.id,
                    entities: kg
                } AS metadata
            """
        )
        retriever = chunk_vector.as_retriever()
        return retriever

 
    def insert_document(self, document_source) -> bool:
        file_path = Path(document_source)  
        if not file_path.exists():
            print(f"File not found: {document_source}")
            return False
        
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
                
            list_of_documents = split_docs
            if len(list_of_documents) > 0:
                for chunk in list_of_documents:
                    filename = os.path.basename(chunk.metadata["source"])
                    chunk_id = f"{filename}.{chunk.metadata['page']}"  # Fixed f-string syntax
                    chunk_embedding = self.embedding.embed_query(chunk_id)
                    properties = {
                        "filename": filename,
                        "chunk_id": chunk_id,
                        "text": chunk.page_content,
                        "embedding": chunk_embedding
                    }
                    self.graph.query("""
                        MERGE (d:Document {id: $filename})
                        MERGE (c:Chunk {id: $chunk_id})
                        SET c.text = $text
                        MERGE (d)<-[:PART_OF]-(c)
                        WITH c
                        CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
                        """,
                        properties
                    )
                    graph_docs = self.doc_transformer.convert_to_graph_documents([chunk])
                    for graph_doc in graph_docs:
                        chunk_node = Node(
                            id=chunk_id,
                            type="Chunk"
                        )
    
                        for node in graph_doc.nodes:
                            graph_doc.relationships.append(
                                Relationship(
                                    source=chunk_node,
                                    target=node,
                                    type="HAS_ENTITY"
                                )
                            )
                    self.graph.add_graph_documents(graph_docs)
                self.graph.query("""
                    CREATE VECTOR INDEX `chunkVector`
                    IF NOT EXISTS
                    FOR (c: Chunk) ON (c.textEmbedding)
                    OPTIONS {indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                    }};"""
                )
                return True
            else:
                print("Error in inserting documents")
                return False
        else:
            print(f"Unsupported file type: {file_path.suffix}")
            return False