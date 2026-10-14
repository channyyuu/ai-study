from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage

load_dotenv()

# 1. 로드 (06의 pymupdf.open + get_text 부분)
docs = PyMuPDFLoader("doc.pdf").load()
print(f"[로드] {len(docs)}페이지")

# 2. 청킹 (06의 while문 부분)
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
chunks = splitter.split_documents(docs)
print(f"[청킹] {len(chunks)}개 조각")

# 3+4. 임베딩 + 저장 (06의 embeddings.create + CREATE TABLE + INSERT 부분)
vector_store = PGVector(
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="ibk_report",
    connection="postgresql+psycopg://postgres:postgres@localhost:5433/postgres",
    pre_delete_collection=True,  # 재실행 시 기존 데이터 삭제
)
vector_store.add_documents(chunks)
print("[저장] 완료")

# 5. 질문 → 검색 → 답변 (07 부분)
model = init_chat_model("openai:gpt-6-luna")

while True:
    question = input("\n질문: ")
    if question == "q":
        break

    results = vector_store.similarity_search_with_score(question, k=3)
    print("\n--- 검색된 조각 ---")
    for doc, score in results:
        print(f"[p.{doc.metadata.get('page', 0) + 1} | 거리 {score:.3f}] {doc.page_content[:80]}...")

    context = "\n\n".join(f"[p.{d.metadata.get('page', 0) + 1}] {d.page_content}" for d, _ in results)
    response = model.invoke([
        SystemMessage("아래 [문서] 내용만 근거로 답해. 문서에 없으면 '문서에서 찾을 수 없습니다'라고 답해. 끝에 참고 페이지를 적어."),
        HumanMessage(f"[문서]\n{context}\n\n[질문]\n{question}"),
    ])
    print("\n--- 답변 ---")
    print(response.text)