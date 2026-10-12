from dotenv import load_dotenv
from openai import OpenAI
# 기본 호출

load_dotenv()     # .env의 OPENAI_API_KEY를 환경변수로 로드
client = OpenAI() # 환경변수에서 키를 자동으로 읽음

response = client.responses.create(
    model="gpt-6-luna",
    # instructions: 시스템 프롬프트(역할 지정)
    instructions="너는 친절한 한국어 조수야. 짧게 답해.", 
    # input: 사용자 질문
    input="MES가 뭔지 한 문장으로 설명해줘",              
)

print(response.output_text)
print("사용 토큰:", response.usage.total_tokens)