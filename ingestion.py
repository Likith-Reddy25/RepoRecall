import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_chroma import Chroma
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

def ingest_repo(repo_path: str="./git_repo", chroma_dir: str="./chroma_db"):
    print(f"Loading files from {repo_path}...")
    
    if not os.path.exists(repo_path):
        raise FileNotFoundError(f"The directory {repo_path} does not exits. Please create or add it")
    
    loader= DirectoryLoader(
        repo_path, 
        glob="**/*.py",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    raw_documents=loader.load()
    print(f"Loaded {len(raw_documents)} documents.")
    
    print("Chunking starts")

    text_splitter= RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=800, chunk_overlap=100
    )
    docs= text_splitter.split_documents(raw_documents)

    for doc in docs:
        full_path= doc.metadata.get("source", "")
        rel_path= os.path.relpath(full_path, repo_path)
        doc.metadata["file_path"]= rel_path

    print(f"Split Code into {len(docs)} logical chunks")

    # Generate Embeddings and store in  ChromaDB
    embeddings= HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store= Chroma.from_documents(documents=docs,
                                        embedding=embeddings,
                                        collection_name="codebase_rag",
                                        persist_directory=chroma_dir)
    
    print("Ingestion Complete!!")
    print(f"Total Vectors stored: {vector_store._collection.count()}")


if __name__=="__main__":
    ingest_repo()






    


