import json
import operator
from typing import Literal
from typing_extensions import TypedDict, Annotated
from datetime import date

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

vector_store = PGVector(
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="ibk_report",
    connection="postgresql+psycopg://postgres:postgres@localhost:5433/postgres",
)

# ── 도구 (05의 tools JSON 설명서 대신 @tool + docstring) ──
@tool
def get_defect_count(line: str, date: str) -> dict:
    """특정 라인의 특정 날짜 불량 건수를 조회한다.

    Args:
        line: 라인명. 1라인 또는 2라인
        date: 조회 날짜. YYYY-MM-DD 형식
    """
    return {"line": line, "date": date, "defect_count": {"1라인": 12, "2라인": 3}.get(line, 0)}

@tool
def get_equipment_status(equipment_id: str) -> dict:
    """설비 ID로 현재 설비 가동 상태를 조회한다.

    Args:
        equipment_id: 설비 ID. 예: EQ0001, EQ0002
    """
    equipment_id = equipment_id.strip().upper()
    status = {"EQ0001": "가동 중", "EQ0002": "정지(금형 교체)"}.get(equipment_id, "알 수 없음")
    return {"equipment_id": equipment_id, "status": status}

@tool
def search_report(query: str) -> list[dict]:
    """IBK증권 반도체 보고서(삼성전자·SK하이닉스)에서 관련 내용을 검색한다.

    Args:
        query: 검색할 내용
    """
    docs = vector_store.similarity_search(query, k=3)
    return [{"page": d.metadata.get("page", 0) + 1, "content": d.page_content} for d in docs]

tools = [get_defect_count, get_equipment_status, search_report]
tools_by_name = {t.name: t for t in tools}
model = init_chat_model("openai:gpt-6-luna").bind_tools(tools)

# ── 상태: 대화 기록 (05의 messages 리스트 역할) ──
class State(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]  # 새 메시지는 기존 리스트에 이어붙임

# ── 노드 1: LLM 판단 ──
def llm_call(state: State):
    response = model.invoke(
        [SystemMessage(f"너는 공장 관리 도우미야. 오늘 날짜는 {date.today()}이야. 필요하면 도구를 써서 답해.")]
        + state["messages"]
    )
    return {"messages": [response]}

# ── 노드 2: 도구 실행 ──
def tool_node(state: State):
    results = []
    for call in state["messages"][-1].tool_calls:
        print(f"  [도구 실행] {call['name']}({call['args']})")
        output = tools_by_name[call["name"]].invoke(call["args"])
        results.append(ToolMessage(content=json.dumps(output, ensure_ascii=False), tool_call_id=call["id"]))
    return {"messages": results}

# ── 분기: 도구 호출 요청이 있으면 tool_node로, 없으면 끝 ──
def should_continue(state: State) -> Literal["tool_node", END]:
    return "tool_node" if state["messages"][-1].tool_calls else END

# ── 그래프 조립 ──
builder = StateGraph(State)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)
builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")

agent = builder.compile(checkpointer=InMemorySaver())  # 대화 상태 저장소, history리스트를 안 만들어도 앞 질문을 기억

print("=== 그래프 구조 (mermaid) ===")
print(agent.get_graph().draw_mermaid())

config = {"configurable": {"thread_id": "chaeyeon-2"}}  # 대화방 번호
while True:
    question = input("\n나: ")
    if question == "q":
        break
    result = agent.invoke({"messages": [HumanMessage(question)]}, config)
    print("AI:", result["messages"][-1].text)