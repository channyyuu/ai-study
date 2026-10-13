import pymupdf
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

PDF_PATH = "doc.pdf"
CHUNK_SIZE = 500     # 한 조각 글자 수(500~1000 권장 / 1000이 국룩)
OVERLAP = 100        # 조각끼리 겹치는 글자 수 (문장이 중간에 잘리는 것 보완, 100~200 권장 / 200이 국룰)
EMBED_MODEL = "text-embedding-3-small"  # 1536차원

# 1. 파싱: PDF → 페이지별 텍스트
doc = pymupdf.open(PDF_PATH)
pages = [(i + 1, page.get_text()) for i, page in enumerate(doc)]
print(f"[파싱] {len(pages)}페이지")

# 2. 청킹: 텍스트 → 일정 길이 조각
chunks = []  # (페이지, 내용)
for page_no, text in pages:
    text = " ".join(text.split())  # 줄바꿈/공백 정리
    start = 0
    while start < len(text):
        chunks.append((page_no, text[start:start + CHUNK_SIZE]))
        start += CHUNK_SIZE - OVERLAP
print(f"[청킹] {len(chunks)}개 조각")

# 3. 임베딩: 조각 → 벡터 (여러 개를 한 번에 요청)
response = client.embeddings.create(model=EMBED_MODEL, input=[c[1] for c in chunks])
embeddings = [d.embedding for d in response.data]
print(f"[임베딩] 벡터 길이 {len(embeddings[0])}")

# 4. 저장: pgvector
with psycopg.connect(
    "host=localhost port=5433 dbname=postgres user=postgres password=postgres",
    autocommit=True,
) as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id          BIGSERIAL PRIMARY KEY,
            source      TEXT,
            page        INT,
            content     TEXT,
            embedding   VECTOR(1536)
        )
    """)
    conn.execute("DELETE FROM documents WHERE source = %s", (PDF_PATH,))  # 재실행 시 중복 방지
    for (page_no, content), emb in zip(chunks, embeddings):
        conn.execute(
            "INSERT INTO documents (source, page, content, embedding) VALUES (%s, %s, %s, %s)",
            (PDF_PATH, page_no, content, np.array(emb)),
        )
print("[저장] 완료")