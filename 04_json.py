from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
# JSON 응답

load_dotenv()
client = OpenAI()

class DefectReport(BaseModel):
    part_name: str
    defect_type: str
    severity: int # 1~5
    action: str

response = client.responses.parse(
    model="gpt-6-luna",
    input="도어 트림 조립 중 스크래치 발견, 꽤 깊음. 라인 정지는 아님. 재작업 필요할 듯",
    text_format=DefectReport,
)

report = response.output_parsed
print(report)
print(report.severity, type(report.severity))