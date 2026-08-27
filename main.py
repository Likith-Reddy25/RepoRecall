import os
import requests
from fastapi import FastAPI, BackgroundTasks, Request
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

app= FastAPI(title= "Codebase RAG Live Sync")

# 1) Initialize Vector store and Splitter
embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store= Chroma(
    collection_name="codebase_rag",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

splitter= RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=800, chunk_overlap= 100
)

# 2) Configuration for Github repo fetch
GITHUB_TOKEN= os.getenv("GITHUB_TOKEN", "")

def process_commit_changes(commits: list, repo_full_name: str, ref: str):
    """Background task to sync modified, added, and delete files from ChromaDB"""
    branch= ref.split("/")[-1]

    for commit in commits:
        # Process Removed files
        for removed_file in commit.get("removed", []):
            if removed_file.endswith(".py"):
                print(f"[Sync] Deleting vectors for removed file: {removed_file}")
                # Filter by meta data field we added during ingestion 
                vector_store._collection.delete(where={"file_path": removed_file})

        # Process Added and Modified Files
        modified_files= commit.get("added",[])+ commit.get("modified", [])
        for file_path in modified_files:
            if file_path.endswith(".py"):
                print(f"[Sync] Updated vectors for modified file: {file_path}")

                # Step 1 : Purge existing chunks for this specific  file
                vector_store._collection.delete(where={"file_path": file_path})

                # Step 2 : Fetch raw updated content directly from Github raw content API
                raw_url= f"https://raw.githubusercontent.com/{repo_full_name}/{file_path}"
                headers={}
                if GITHUB_TOKEN:
                    headers["Authorization"]=f"token {GITHUB_TOKEN}"
                
                res= requests.get(raw_url, headers=headers)
                if res.status_code==200:
                    code_content= res.text

                    #Step 3: Split into chunks
                    chunks= splitter.create_documents(
                        [code_content], metadatas=[{"file_path":file_path}]
                    )
                    
                    # Step 4: Put the new chunks into chromaDB
                    vector_store.add_documents(chunks)
                    print(f"[Sync] Successfully indexed {len(chunks)} chunks for {file_path}")
                else:
                    print(f"Failed to fetch raw file: {file_path} (Status {res.status_code})")


@app.post("/webhook")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    """Receives Github push event payloads and queues live vector updates"""
    payload= await request.json()

    # Verify it's a push event with commit history
    commits= payload.get("commits", [])
    repository=payload.get("repository",{})
    repo_full_name= repository.get("full_name", "")
    ref= payload.get("ref", "refs/heads/main")

    if commits and repo_full_name:
        # Queue processing in background so webhook responds with 200 immediately
        background_tasks.add_task(process_commit_changes, commits, repo_full_name, ref)
        return {"status": "processing", "commits_count": len(commits)}
    
    return {"status": "ignored", "reason":"No commit payload present"}


if __name__=="__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload= True)



