from pathlib import Path
import numpy as np
import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg import register_vector
from mcp.server.mcpserver import MCPServer
# 05_tool.py와 비교 시 tools JSON설명서가 사라짐
# 함수의 타입 힌트와 docstring을 보고 SDK가 설명서를 자동으로 생성
# 표준입출력(stdio)으로 클라이언트와 통신하기 때문에, print가 섞이면 통신 메시지가 깨짐

#05와 비교: 05에선 도구가 내 프로그램 안에 묶여 있어서 그 프로그램에서만 쓸 수 있음. MCP 서버로 분리하니 코드 한 줄 안 고치고 Claude Desktop이 그대로 가져다 씀. 그래서 흔히 “AI용 USB-C”라고 비유.
#구조: Claude Desktop = MCP 클라이언트(호스트), mes_server.py = MCP 서버. 클라이언트가 서버에 "무슨 도구 있어?" 물어보고 목록을 받아 LLM에게 넘김.
#통신 방식: 지금은 같은 PC에서 프로세스로 띄우는 stdio 방식이고, 원격 서버로 띄울 땐 HTTP(Streamable HTTP) 방식 사용.

# Claude Desktop이 서버를 다른 폴더에서 실행할 수 있어서, .env 위치를 파일 기준으로 지정
load_dotenv(Path(__file__).parent / ".env")
openai_client = OpenAI()

DB_DSN = "host=localhost port=5433 dbname=postgres user=postgres password=postgres"

mcp = MCPServer("mes")

# TEST 데이터 (실무 시 DB 조회)
DEFECT_DATA = {"1라인": 12, "2라인": 3}
EQUIPMENT_DATA = {"EQ0001": "가동 중", "EQ0002": "정지(금형 교체)"}


@mcp.tool()
def get_defect_count(line: str, date: str) -> dict:
    """특정 라인의 특정 날짜 불량 건수를 조회한다.

    Args:
        line: 라인명. 1라인 또는 2라인
        date: 조회 날짜. YYYY-MM-DD 형식
    """
    return {"line": line, "date": date, "defect_count": DEFECT_DATA.get(line, 0)}


@mcp.tool()
def get_equipment_status(equipment_id: str) -> dict:
    """설비 ID로 현재 설비 가동 상태를 조회한다.

    Args:
        equipment_id: 설비 ID. 예: EQ0001, EQ0002
    """
    equipment_id = equipment_id.strip().upper()  # 소문자로 입력했을 경우에도 처리
    return {
        "equipment_id": equipment_id,
        "status": EQUIPMENT_DATA.get(equipment_id, "알 수 없음"),
    }


@mcp.tool()
def search_report(query: str, top_k: int = 5) -> list[dict]:
    """IBK증권 반도체 보고서(삼성전자·SK하이닉스 최근 동향)에서 질문과 관련된 내용을 검색한다.
    실적 추정치, 업황, 메모리 가격 전망, 투자의견 등 보고서 내용이 필요할 때 사용한다.

    Args:
        query: 검색할 내용. 예: 삼성전자 4분기 영업이익 추정
        top_k: 가져올 문서 조각 수 (기본 5)
    """
    # 1. 질문 임베딩
    q_emb = openai_client.embeddings.create(
        model="text-embedding-3-small", input=query
    ).data[0].embedding

    # 2. pgvector에서 가까운 조각 검색
    with psycopg.connect(DB_DSN) as conn:
        register_vector(conn)
        rows = conn.execute(
            """
            SELECT page, content, 1 - (embedding <=> %s) AS similarity
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (np.array(q_emb), np.array(q_emb), top_k),
        ).fetchall()

    # 3. 검색 결과만 돌려줌 (답변 생성은 Claude가 함)
    return [
        {"page": page, "similarity": round(float(sim), 3), "content": content}
        for page, content, sim in rows
    ]


if __name__ == "__main__":
    mcp.run()