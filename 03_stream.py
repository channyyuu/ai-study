from dotenv import load_dotenv
from openai import OpenAI
# 스트리밍
# 답변은 조각(delta) 단위로 실시간 수신. ChatGPT 화면에서 글자가 타이핑되듯 나오는 게 이 방식이고, 사용자가 체감하는 대기 시간을 줄이는 목적.
# 웹에서는 보통 SSE(Server-Sent Events)로 프론트에 전달.

load_dotenv()
client = OpenAI()

stream = client.responses.create(
    model="gpt-6-luna",
    input="스마트팩토리의 장점을 5가지 설명해줘",
    stream=True,
)

for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
print()