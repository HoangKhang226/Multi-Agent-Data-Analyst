from src.retrieval.vector_db import VectorDBManager
from tests.mocks.fake_embedding import FakeEmbedding


def test_qdrant_add_and_retrieve(tmp_path):
    embedding = FakeEmbedding()

    db = VectorDBManager(embedding_model=embedding)

    # reset DB để clean test
    db.reset_db()

    class Node:
        def __init__(self, text, id):
            self.text = text
            self.node_id = id

    nodes = [
        Node("AI is powerful", "1"),
        Node("RAG improves retrieval", "2")
    ]

    db.add_documents(nodes, collection_name="test_collection")

    retriever = db.get_retriever(collection_name="test_collection")

    results = retriever.retrieve("What is RAG?")

    assert len(results) > 0