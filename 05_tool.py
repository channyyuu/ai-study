import json
from datetime import date
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# 1. 실제로 실행될 함수 (지금은 TEST 데이터, 실무 시 DB 조회)
def get_defect_count(line: str, date: str) -> dict:
    fake = {"1라인": 12, "2라인": 3}
    return {"line": line, "date": date, "defect_count": fake.get(line, 0)}

def get_equipment_status(equipment_id: str) -> dict:
    fake = {"EQ0001": "가동 중", "EQ0002": "정지(금형 교체)"}
    return {"equipment_id": equipment_id, "status": fake.get(equipment_id, "알 수 없음")}

FUNCTIONS = {
    "get_defect_count": get_defect_count,
    "get_equipment_status": get_equipment_status,
}

# 2. LLM에게 보여줄 "함수 설명서"
tools = [
    {
        "type": "function",
        "name": "get_defect_count",
        "description": "특정 라인의 특정 날짜 불량 건수를 조회한다",
        "parameters": {
            "type": "object",
            "properties": {
                "line": {"type": "string", "description": "라인명. 예: 1라인"},
                "date": {"type": "string", "description": "YYYY-MM-DD 형식"},
            },
            "required": ["line", "date"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_equipment_status",
        "description": "설비 ID로 현재 설비 가동 상태를 조회한다",
        "parameters": {
            "type": "object",
            "properties": {
                "equipment_id": {"type": "string", "description": "설비 ID. 예: EQ0001"},
            },
            "required": ["equipment_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

question = input("질문: ")
messages = [{"role": "user", "content": question}]

# 3. LLM이 함수 호출을 그만 요청할 때까지 반복
while True:
    response = client.responses.create(
        model="gpt-6-luna",
        instructions=f"너는 공장 관리 도우미야. 오늘 날짜는 {date.today()}이야.",
        input=messages,
        tools=tools,
    )

    calls = [item for item in response.output if item.type == "function_call"]
    if not calls:
        print("AI:", response.output_text)
        break

    messages += response.output  # LLM의 호출 요청도 기록에 남김
    for call in calls:
        args = json.loads(call.arguments)
        print(f"  [LLM 요청] {call.name}({args})")
        result = FUNCTIONS[call.name](**args)   # 실제 실행은 내 코드가
        print(f"  [실행 결과] {result}")
        messages.append({
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": json.dumps(result, ensure_ascii=False),
        })