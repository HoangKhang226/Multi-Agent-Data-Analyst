from tests.mocks.fake_llm import FakeLLM
from tests.mocks.fake_embedding import FakeEmbedding
from src.retrieval.vector_db import VectorDBManager


def test_full_multi_agent_flow():
    llm = FakeLLM()
    embedding = FakeEmbedding()

    db = VectorDBManager(embedding_model=embedding)
    db.reset_db()

    class Node:
        def __init__(self, text, id):
            self.text = text
            self.node_id = id

    # ingest
    nodes = [Node("RAG is retrieval augmented generation", "1")]
    db.add_documents(nodes, "agent_test")

    # router
    decision = llm.invoke("read this document")

    if "RAG" in decision:
        retriever = db.get_retriever("agent_test")
        results = retriever.retrieve("What is RAG?")
        output = results[0].text
    else:
        output = "fallback"

    assert "RAG" in output