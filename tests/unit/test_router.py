from tests.mocks.fake_llm import FakeLLM


def simple_router(llm, query: str):
    decision = llm.invoke(query)

    if "PANDAS" in decision:
        return "pandas"
    elif "RAG" in decision:
        return "rag"
    return "general"


def test_router_pandas():
    llm = FakeLLM()
    route = simple_router(llm, "analyze this csv data")
    assert route == "pandas"


def test_router_rag():
    llm = FakeLLM()
    route = simple_router(llm, "read this document")
    assert route == "rag"