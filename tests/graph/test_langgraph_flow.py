from tests.mocks.fake_llm import FakeLLM


def router_node(state):
    llm = FakeLLM()
    decision = llm.invoke(state["input"])

    if "PANDAS" in decision:
        return {"route": "pandas"}
    elif "RAG" in decision:
        return {"route": "rag"}
    return {"route": "general"}


def pandas_node(state):
    return {"output": "data analysis done"}


def rag_node(state):
    return {"output": "retrieved knowledge"}


def general_node(state):
    return {"output": "general answer"}


def test_langgraph_flow():
    state = {"input": "analyze csv data"}

    route = router_node(state)

    if route["route"] == "pandas":
        result = pandas_node(state)
    elif route["route"] == "rag":
        result = rag_node(state)
    else:
        result = general_node(state)

    assert "output" in result