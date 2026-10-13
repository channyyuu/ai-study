import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

TOP_K = 3

conn = psycopg.connect("host=localhost port=5433 dbname=postgres user=postgres password=postgres")
register_vector(conn)

while True:
    question = input("\n질문: ")
    if question == "q":
        break

    # 1. 질문도 같은 모델로 임베딩
    q_emb = client.embeddings.create(model="text-embedding-3-small", input=question).data[0].embedding

    # 2. 의미가 가장 가까운 조각 검색 (<=> : 코사인 거리)
    rows = conn.execute(
        """
        SELECT page, content, 1 - (embedding <=> %s) AS similarity
        FROM documents
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (np.array(q_emb), np.array(q_emb), TOP_K),
    ).fetchall()

    print("\n--- 검색된 조각 ---")
    for page, content, sim in rows:
        print(f"[p.{page} | 유사도 {sim:.3f}] {content[:80]}...")

    # 3. 검색 결과를 근거로 LLM에게 답변 요청
    context = "\n\n".join(f"[p.{page}] {content}" for page, content, _ in rows)
    response = client.responses.create(
        model="gpt-6-luna",
        instructions=(
            "아래 [문서] 내용만 근거로 답해. 문서에 없는 내용이면 '문서에서 찾을 수 없습니다'라고 답해. "
            "답변 끝에 참고한 페이지 번호를 적어."
        ),
        input=f"[문서]\n{context}\n\n[질문]\n{question}",
    )
    print("\n--- 답변 ---")
    print(response.output_text)