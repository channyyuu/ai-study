from dotenv import load_dotenv
from openai import OpenAI
# 대화 이어가기
# LLM은 상태가 없다(stateless). 기억하는 것처럼 보이는 건 매번 이전 대화를 통째로 다시 보내기 때문이고, 대화가 길어질수록 토큰(비용) 증가.

load_dotenv()
client = OpenAI()

history = []  # 대화 기록을 내가 직접 들고 있음

while True:
    user = input("YOU: ")
    if user == "q":
        break
    history.append({"role": "user", "content": user})

    response = client.responses.create(model="gpt-6-luna", input=history)
    answer = response.output_text
    print("AI:", answer)

    # 이 부분을 주석 처리 하여 기억 못 하는 것까지 확인 필요
    history.append({"role": "assistant", "content": answer})