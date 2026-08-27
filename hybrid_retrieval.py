from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

embedding_model= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store= Chroma(
    collection_name="codebase_rag",
    embedding_function= embedding_model,
    persist_directory="./chroma_db"
)

# llm= GoogleGenerativeAI(model="gemini-3.5-flash-lite")
llm= ChatGroq(model="openai/gpt-oss-20b")


# Initialise the local Flashrank model
ranker= Ranker(model_name="ms-marco-MiniLM-L-12-v2")

def hybrid_code_search(query: str, top_k: int = 5):
    print(f"\nSearching for '{query}'")

    # ---------------------------------------------------------
    # STEP 1: Dense Vector Search
    # ---------------------------------------------------------
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 15}
    )

    vector_results = retriever.invoke(query)

    if not vector_results:
        print("Found no relevant documents")
        return []

    # ---------------------------------------------------------
    # STEP 2: Prepare documents
    # ---------------------------------------------------------
    candidates_docs = [doc.page_content for doc in vector_results]
    candidates_meta = [doc.metadata for doc in vector_results]

    # ---------------------------------------------------------
    # STEP 3: BM25 Keyword Search
    # ---------------------------------------------------------

    # Tokenize each code chunk
    tokenized_corpus = [
        doc.split() for doc in candidates_docs
    ]

    # Create BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    # Tokenize user's query
    tokenized_query = query.split()

    # Calculate BM25 score for every candidate
    bm25_scores = bm25.get_scores(tokenized_query)

    # ---------------------------------------------------------
    # STEP 4: Normalize BM25 scores
    # ---------------------------------------------------------
    max_bm25 = max(bm25_scores)

    if max_bm25 > 0:
        normalized_bm25 = [
            score / max_bm25
            for score in bm25_scores
        ]
    else:
        normalized_bm25 = [0] * len(bm25_scores)

    # ---------------------------------------------------------
    # STEP 5: Create hybrid candidates
    # ---------------------------------------------------------

    passages = []

    for idx, doc in enumerate(candidates_docs):

        # Dense score isn't returned by retriever.invoke(),
        # so for this simple implementation we use BM25
        # to enhance the candidate ordering.

        passages.append({
            "id": idx,
            "text": doc,
            "meta": candidates_meta[idx],
            "bm25_score": normalized_bm25[idx]
        })

    # ---------------------------------------------------------
    # STEP 6: Sort using BM25
    # ---------------------------------------------------------

    passages.sort(
        key=lambda x: x["bm25_score"],
        reverse=True
    )

    # Keep the best candidates for FlashRank
    passages_for_reranking = passages[:15]

    # ---------------------------------------------------------
    # STEP 7: FlashRank Reranking
    # ---------------------------------------------------------

    rerank_passages = []

    for item in passages_for_reranking:
        rerank_passages.append({
            "id": item["id"],
            "text": item["text"],
            "meta": item["meta"]
        })

    rerank_request = RerankRequest(
        query=query,
        passages=rerank_passages
    )

    reranked_results = ranker.rerank(
        rerank_request
    )

    # ---------------------------------------------------------
    # STEP 8: Return top K
    # ---------------------------------------------------------

    top_results = reranked_results[:top_k]

    return top_results

def answer_code_question(query: str):
    # Retrieve the valid chunks
    retrieved_chunks= hybrid_code_search(query, top_k=3)

    if not retrieved_chunks:
        return "No relevant code found in the repository."
    
    context_str = ""
    for idx, item in enumerate(retrieved_chunks, 1):
        file_path = item["meta"].get("file_path", "Unknown File")
        context_str += f"\n--- Code Snippet {idx} (File: {file_path}) ---\n"
        context_str += f"{item['text']}\n"

    # 3. Prompt the LLM
    prompt = f"""You are an expert developer assistant. Answer the user's question using ONLY the retrieved code snippets below.
If the answer cannot be found in the context, state that clearly.

Retrieved Code Context:
{context_str}

User Question: {query}

Answer:"""

    response = llm.invoke(prompt)
    return response.content


if __name__=="__main__":
    test_query="How does dependency injection work in FastAPI routes?"
    print(f"Question: {test_query}\n")

    answer= answer_code_question(test_query)
    print("--LLM answer--")
    print(answer)