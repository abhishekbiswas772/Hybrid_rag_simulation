from documet_igestion_chroma import ChromaDocumentManager

pdf_path = "/home/abhishek/Desktop/hybrid_rag/Abhishek Resume.pdf"
chroma_docs = ChromaDocumentManager()
print(chroma_docs.ingest_document(document_source=pdf_path))
retriver = chroma_docs.get_retriver()
print(retriver.invoke("experince"))