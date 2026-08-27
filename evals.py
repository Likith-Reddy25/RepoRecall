import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from langchain_groq import ChatGroq
from hybrid_retrieval import hybrid_code_search

load_dotenv()

# 🔹 LLM (same as your RAG)
llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0
)

# 🔹 Embedding model (LOCAL, FREE)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 Test dataset
test_cases = [
    {
        "question": "How is HTTPException defined or handled?",
        "ground_truth": "HTTPException is defined as an exception class that can be raised to return HTTP error responses.",
        "relevant_keywords": ["HTTPException", "error", "exception"]
    },
    {
        "question": "How do background tasks work in responses?",
        "ground_truth": "BackgroundTasks class allows adding tasks to be executed after returning a response",
        "relevant_keywords": ["BackgroundTasks", "tasks", "response"]
    },
]

# =========================
# 🔍 METRICS
# =========================

def embedding_similarity(answer, ground_truth):
    a_emb = embedder.encode([answer])
    gt_emb = embedder.encode([ground_truth])
    return cosine_similarity(a_emb, gt_emb)[0][0]


def keyword_recall(contexts, keywords):
    context_text = " ".join(contexts).lower()
    hits = sum(1 for k in keywords if k.lower() in context_text)
    return hits / len(keywords)


import re

def llm_judge(question, answer, ground_truth):
    prompt = f"""
You are evaluating a RAG system.

Question: {question}
Answer: {answer}
Ground Truth: {ground_truth}

Give a score between 0 and 1 for correctness.

Return ONLY a number like:
0.0
0.5
0.8
1.0
"""

    response = llm.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    # 🔥 Extract number safely
    match = re.search(r"\d+(\.\d+)?", text)

    if match:
        return float(match.group())
    
    return 0.0


# =========================
# 🚀 MAIN EVALUATION
# =========================

def run_eval():
    print("Running Custom RAG Evaluation...\n")

    total_sim = 0
    total_recall = 0
    total_llm_score = 0

    for i, item in enumerate(test_cases):
        q = item["question"]
        gt = item["ground_truth"]
        keywords = item["relevant_keywords"]

        print(f"\n--- Test Case {i+1} ---")
        print("Question:", q)

        # 🔍 Retrieval
        retrieved_chunks = hybrid_code_search(q, top_k=3)
        contexts = [c["text"] for c in retrieved_chunks]

        context_str = "\n".join(contexts)

        # 🤖 Generate answer
        prompt = f"""
Answer ONLY using the given context.
If answer is not found, say "Not found".

Context:
{context_str}

Question: {q}
"""

        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        print("Answer:", answer)

        # 📊 Metrics
        sim = embedding_similarity(answer, gt)
        recall = keyword_recall(contexts, keywords)
        judge = llm_judge(q, answer, gt)

        total_sim += sim
        total_recall += recall
        total_llm_score += judge

        print(f"Embedding Similarity: {sim:.3f}")
        print(f"Context Recall: {recall:.3f}")
        print(f"LLM Judge Score: {judge:.3f}")

    n = len(test_cases)

    print("\n===== FINAL SCORES =====")
    print(f"Avg Embedding Similarity: {total_sim/n:.3f}")
    print(f"Avg Context Recall: {total_recall/n:.3f}")
    print(f"Avg LLM Judge Score: {total_llm_score/n:.3f}")


if __name__ == "__main__":
    run_eval()