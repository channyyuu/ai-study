from dotenv import load_dotenv
from openai import OpenAI

# pydantic
# 데이터 유효성 검사 및 설정 관리를 위한 라이브러리
# FastAPI와 같은 웹 프레임워크와 함께 사용되며, 데이터를 구조화하고 검증하는 데 유용
# BaseModel은 Pydantic의 핵심 클래스 중 하나로, 데이터 모델을 정의하는 데 사용
from pydantic import BaseModel 

# JSON 응답

load_dotenv()
client = OpenAI()

class DefectReport(BaseModel):
    part_name: str # 부위명
    defect_type: str # 결함명
    severity: int # 심각성, 1~5
    action: str # 방안

response = client.responses.parse(
    model="gpt-6-luna",
    input="도어 트림 조립 중 스크래치 발견, 꽤 깊음. 라인 정지는 아님. 재작업 필요할 듯",
    text_format=DefectReport,
)

report = response.output_parsed
print(report)
print(report.severity, type(report.severity))