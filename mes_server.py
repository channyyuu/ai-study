from mcp.server.mcpserver import MCPServer
# 05_tool.py와 비교 시 tools JSON설명서가 사라짐
# 함수의 타입 힌트와 docstring을 보고 SDK가 설명서를 자동으로 생성
# 표준입출력(stdio)으로 클라이언트와 통신하기 때문에, print가 섞이면 통신 메시지가 깨짐

#05와 비교: 05에선 도구가 내 프로그램 안에 묶여 있어서 그 프로그램에서만 쓸 수 있음. MCP 서버로 분리하니 코드 한 줄 안 고치고 Claude Desktop이 그대로 가져다 씀. 그래서 흔히 “AI용 USB-C”라고 비유.
#구조: Claude Desktop = MCP 클라이언트(호스트), mes_server.py = MCP 서버. 클라이언트가 서버에 "무슨 도구 있어?" 물어보고 목록을 받아 LLM에게 넘김.
#통신 방식: 지금은 같은 PC에서 프로세스로 띄우는 stdio 방식이고, 원격 서버로 띄울 땐 HTTP(Streamable HTTP) 방식 사용.

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
    equipment_id = equipment_id.strip().upper() # 소문자로 입력했을 경우에도 처리
    return {
        "equipment_id": equipment_id,
        "status": EQUIPMENT_DATA.get(equipment_id, "알 수 없음"),
    }


if __name__ == "__main__":
    mcp.run()