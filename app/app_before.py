%%writefile app_before.py

import os

import streamlit as st
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# 페이지 설정
st.set_page_config(page_title="Before - 항공권 환불 챗봇", layout="wide")
st.title("✈️ Before 버전 - 환불 상담 챗봇")
st.markdown("### 🔴 문제 상황 재현 버전")
st.caption("chunk_size=800, overlap=100, 동의어 없음")

# 사이드바
with st.sidebar:
    st.header("📚 Before 설정")
    st.info("⚠️ 이 버전은 초기 상태를 재현합니다")

    # API 키 입력
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    k = st.slider("검색 개수 k", 1, 5, 3)
    show_sources = st.checkbox("근거 표시", value=True)

    st.divider()

    st.header("❓ 테스트 질문")
    test_questions = [
        "제주항공 노쇼 위약금은?",
        "진에어 국내선 노쇼 위약금은 얼마인가요?",
        "no-show penalty",
        "대한항공 국제선 환불 수수료 표",
        "제주항공 FLEX 운임 환불"
    ]

    for q in test_questions:
        if st.button(q, key=q):
            st.session_state["question"] = q

    st.divider()
    if st.button("대화 초기화"):
        st.session_state.pop("messages", None)
        st.rerun()

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "question" not in st.session_state:
    st.session_state["question"] = None

# API 키 확인
if not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️ OpenAI API 키를 사이드바에 입력하세요")
    st.stop()

# LLM 초기화
@st.cache_resource
def init_llm():
    return ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY")
    )

llm = init_llm()

# 프롬프트
rag_prompt = ChatPromptTemplate.from_template("""
너는 항공권 환불 및 변경을 도와주는 친절한 한국어 상담 챗봇이야.
아래 항공사 정책 문서를 참고해서 질문에 정확하고 친절하게 답변해줘.

정책문서: {context}
사용자 질문: {q}

답변 작성 시 유의사항:
1. 문서에 있는 정보를 기반으로 답변하되, 이해하기 쉽게 설명해줘
2. 환불/변경 수수료, 기간 등 구체적인 정보를 명확히 제시해줘
3. 답변 마지막에 "⚠️ 정확한 정보는 항공사 공식 웹사이트를 확인해주세요"라고 안내해줘

답변:
""")

rag_chain = rag_prompt | llm | StrOutputParser()

# 벡터 DB 초기화
@st.cache_resource
def initialize_vectordb():
    """Before 설정으로 벡터 DB 초기화"""

    # MD 파일 경로 (Corrected path)
    md_path = Path("/content/before")

    # MD 파일 로드
    docs = []
    for md_file in md_path.glob("*.md"):
        try:
            loader = TextLoader(str(md_file), encoding='utf-8')
            file_docs = loader.load()

            for doc in file_docs:
                doc.metadata['source'] = md_file.name

            docs.extend(file_docs)
        except Exception as e:
            st.warning(f"파일 로드 실패: {md_file.name}")
            continue

    if not docs:
        st.error("MD 파일을 찾을 수 없습니다")
        st.stop()

    # Before 설정으로 청킹
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Before: 작은 청크
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    # 벡터 DB 생성
    embeddings = OpenAIEmbeddings(
        model='text-embedding-3-small',
        api_key=os.environ.get("OPENAI_API_KEY")
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="before_chatbot"
    )

    return db, len(docs), len(chunks)

with st.spinner("🔄 벡터 DB 초기화 중..."):
    db, num_docs, num_chunks = initialize_vectordb()

st.sidebar.success(f"✅ {num_docs}개 문서, {num_chunks}개 청크")

# 라우팅 키워드 (Before 버전 - 동의어 없음)
RAG_KEYWORDS = [
    "환불", "교환", "반품", "정책", "항공권", "변경", "취소", "운임",
    "FLEX", "SAVE", "국내선", "국제선", "수수료", "기간", "No-Show",
    "노쇼", "패키지", "처리", "유효기간"
]

def route_to_rag(q):
    """라우팅 판단"""
    ql = q.lower()
    return any(kw.lower() in ql for kw in RAG_KEYWORDS)

def refund_rag(question, k_val):
    """RAG 검색 및 답변"""
    results = db.similarity_search_with_relevance_scores(question, k=k_val)

    if not results:
        return "❌ 관련 규정을 찾지 못했습니다.", []

    context = "\n\n".join([doc.page_content for doc, score in results])
    answer = rag_chain.invoke({'context': context, 'q': question})

    sources = []
    for doc, score in results:
        sources.append({
            'filename': doc.metadata.get('source', 'unknown'),
            'score': score,
            'content': doc.page_content[:150]
        })

    return answer, sources

# 채팅 UI
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 입력 처리
if st.session_state["question"]:
    question = st.session_state["question"]
    st.session_state["question"] = None
else:
    question = st.chat_input("질문을 입력하세요")

if question:
    # 사용자 메시지
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state["messages"].append({"role": "user", "content": question})

    # 어시스턴트 답변
    with st.chat_message("assistant"):
        if route_to_rag(question):
            with st.spinner("🔍 검색 중..."):
                answer, sources = refund_rag(question, k)

            st.markdown(answer)

            # 검색 결과 표시
            if sources:
                st.success(f"✅ {len(sources)}개 문서 검색 완료")
            else:
                st.error("❌ 검색 실패")

            if show_sources and sources:
                with st.expander("🔍 참고 문서"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**{i}. {src['filename']}** (유사도: {src['score']:.2f})")
                        st.caption(src['content'][:100] + "...")
                        st.divider()
        else:
            answer = "죄송합니다. 항공권 환불/변경 관련 질문이 아닌 것 같습니다."
            st.markdown(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})

# 하단 정보
st.markdown("---")
st.caption("🔴 Before 버전 - 문제 상황 재현")
st.caption("특징: chunk_size=800, 동의어 사전 없음, 기본 RAG")
