"""Debug script — inspect what's actually being retrieved."""
from app import load_index

index = load_index()

# Test 1: raw retrieval (no LLM)
retriever = index.as_retriever(similarity_top_k=3)
nodes = retriever.retrieve("What does NRC consider a microreactor?")

print("=" * 70)
print("RETRIEVED CHUNKS")
print("=" * 70)
for i, node in enumerate(nodes, 1):
    print(f"\n--- Chunk {i} (score: {node.score:.3f}) ---")
    print(f"Metadata: {node.metadata}")
    print(f"Text preview: {node.text[:400]}")
    print(f"[Total text length: {len(node.text)} chars]")

# Test 2: pure query engine (no chat history, no condensing)
print("\n" + "=" * 70)
print("QUERY ENGINE ANSWER")
print("=" * 70)
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("What does NRC consider a microreactor?")
print(response)