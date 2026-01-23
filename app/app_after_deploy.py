"""
After 버전 항공권 환불 상담 RAG 챗봇 - Streamlit Cloud 배포 전용

⚠️ 중요: 이 파일은 Streamlit Cloud 배포 전용입니다!
개발/테스트는 app_after.py를 사용하세요.

차이점:
- Colab 매직 커맨드 제거 (%%writefile)
- 상대 경로 사용 (data/after/*.md)
- glob 패턴으로 파일 검색
- 에러 메시지 강화

개발 환경: app_after.py (Colab 호환)
배포 환경: app_after_deploy.py (Streamlit Cloud)

---
기술 스펙:
- chunk_size: 2000
- chunk_overlap: 400
- 동의어 사전: 50+
- 대한항공 통합 (6개 md 파일)
"""

import os
import glob
import streamlit as st
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# 페이지 설정
st.set_page_config(
    page_title="After - 항공권 환불 챗봇",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ After 버전 - 환불 상담 챗봇")
st.markdown("### 🟢 최종 개선 버전 (배포)")
st.caption("chunk_size=2000, overlap=400, 동의어 50+")

# 동의어 사전 (After 핵심 개선)
SYNONYM_DICT = {
    # 노쇼 관련
    "노쇼": ["노쇼", "No-Show", "no-show", "노 쇼", "미탑승", "예약부도"],
    "no-show": ["노쇼", "No-Show", "no-show", "미탑승", "예약부도"],
    "예약부도": ["노쇼", "예약부도", "no-show", "미탑승"],

    # 환불 관련
    "환불": ["환불", "refund", "반환", "취소환불"],
    "refund": ["환불", "refund", "반환"],

    # 변경 관련
    "변경": ["변경", "change", "수정", "교환"],
    "change": ["변경", "change", "수정"],

    # 수수료 관련
    "수수료": ["수수료", "fee", "요금", "비용", "charge", "위약금", "패널티", "penalty"],
    "fee": ["수수료", "fee", "요금", "비용", "charge", "위약금", "패널티", "penalty"],
    "위약금": ["위약금", "패널티", "penalty", "수수료", "fee"],
    "penalty": ["위약금", "패널티", "penalty", "수수료"],

    # 취소 관련
    "취소": ["취소", "cancel", "cancellation", "해지"],
    "cancel": ["취소", "cancel", "cancellation"],

    # 운임 종류
    "특가": ["특가", "특가운임", "프로모션", "promotion", "special"],
    "할인": ["할인", "할인운임", "discount", "세일", "sale"],
    "일반": ["일반", "일반운임", "정상", "정상운임", "normal", "regular"],

    # 운임 등급
    "베이직": ["베이직", "BASIC", "Basic", "basic"],
    "basic": ["베이직", "BASIC", "Basic"],
    "스탠다드": ["스탠다드", "STANDARD", "Standard", "standard"],
    "standard": ["스탠다드", "STANDARD", "Standard"],
    "플렉스": ["플렉스", "FLEX", "Flex", "flex", "flexible"],
    "flex": ["플렉스", "FLEX", "Flex", "flexible"],
    "세이버": ["세이버", "SAVER", "Saver", "saver"],
    "saver": ["세이버", "SAVER", "Saver"],

    # 노선 관련
    "국내선": ["국내선", "domestic", "국내"],
    "domestic": ["국내선", "domestic"],
    "국제선": ["국제선", "international", "국제", "해외"],
    "international": ["국제선", "international"],

    # 탑승수속 관련
    "탑승수속": ["탑승수속", "체크인", "check-in", "수속"],
    "체크인": ["탑승수속", "체크인", "check-in"],
    "check-in": ["탑승수속", "체크인", "check-in"],

    # 게이트 관련
    "게이트": ["게이트", "gate", "출구장"],
    "gate": ["게이트", "gate"],

    # 미탑승
    "미탑승": ["미탑승", "no-show", "미승선", "불탑승"]
}

def expand_query_with_synonyms(query: str) -> str:
    """동의어를 활용한 쿼리 확장"""
    words = query.split()
    expanded_terms = []

    for word in words:
        word_lower = word.lower()
        if word_lower in SYNONYM_DICT:
            synonyms = SYNONYM_DICT[word_lower]
            expanded_terms.extend(synonyms[:3])
        expanded_terms.append(word)

    return " ".join(expanded_terms)

# 사이드바
with st.sidebar:
    st.header("📚 After 설정")
    st.success("✅ 최종 개선 버전")
    st.info("🚀 Streamlit Cloud 배포 버전")

    # API 키 입력
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Streamlit Cloud Secrets에 설정하거나 여기에 입력하세요"
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    k = st.slider("검색 개수 k", 1, 5, 3)
    show_sources = st.checkbox("근거 표시", value=True)
    use_synonyms = st.checkbox("동의어 확장", value=True)

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

    # 동의어 정보 표시
    st.divider()
    with st.expander("🔍 동의어 사전 정보"):
        st.caption(f"총 {len(SYNONYM_DICT)}개 키워드")
        st.caption("예: 노쇼 → no-show, 예약부도")

    st.divider()
    with st.expander("ℹ️ 버전 정보"):
        st.caption("**배포 버전**: app_after_deploy.py")
        st.caption("**개발 버전**: app_after.py (Colab)")
        st.caption("**Chunk Size**: 2000")
        st.caption("**동의어**: 50+")

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "question" not in st.session_state:
    st.session_state["question"] = None

# API 키 확인
if not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️ OpenAI API 키를 설정하세요")
    st.info("""
    **Streamlit Cloud 배포 시:**
    1. Settings → Secrets 메뉴
    2. 다음 내용 입력:
    ```
    OPENAI_API_KEY = "sk-..."
    ```
    
    **로컬 테스트 시:**
    사이드바에서 API 키 입력
    """)
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
3. 표(테이블) 형식의 정보가 있다면 **전체 내용**을 빠짐없이 포함해줘
4. 운임 등급별 차이가 있다면 명확히 구분해서 설명해줘
5. 답변 마지막에 "⚠️ 정확한 정보는 항공사 공식 웹사이트를 확인해주세요"라고 안내해줘

답변:
""")

rag_chain = rag_prompt | llm | StrOutputParser()

# 벡터 DB 초기화
@st.cache_resource
def initialize_vectordb():
    """
    After 설정으로 벡터 DB 초기화
    Streamlit Cloud 배포 최적화 버전
    """

    # MD 파일 경로 패턴 (우선순위 순)
    patterns = [
        "data/after/*.md",       # Streamlit Cloud (상대 경로)
        "./data/after/*.md",     # 로컬 실행
        "../data/after/*.md",    # app/ 폴더에서 실행 시
    ]

    # 파일 검색
    seen = set()
    loader_files = []

    for pat in patterns:
        for fp in glob.glob(pat, recursive=True):
            if fp.endswith(".md") and fp not in seen and Path(fp).is_file():
                seen.add(fp)
                loader_files.append(fp)

    # 로드 결과 표시
    st.sidebar.caption(f"📄 로드된 MD 파일: {len(loader_files)}개")
    
    if loader_files:
        with st.sidebar.expander("📂 로드된 파일 목록", expanded=False):
            for fp in sorted(loader_files):
                filename = Path(fp).name
                # 항공사명 추출
                airline = filename.split('_')[0] if '_' in filename else filename
                st.text(f"✅ {airline}")

    if not loader_files:
        st.error("❌ MD 파일을 찾을 수 없습니다")
        st.info("""
        ## 📂 프로젝트 구조 확인
        
        **필요한 구조:**
        ```
        RAG_Chatbot_v2/
        ├── app/
        │   └── app_after_deploy.py  ← 이 파일
        └── data/
            └── after/               ← MD 파일 위치
                ├── 제주항공_환불규정.md
                ├── 진에어_환불규정.md
                ├── 대한항공_환불규정.md
                ├── 아시아나_환불규정.md
                ├── 이스타항공_환불규정.md
                └── 에어서울_환불규정.md
        ```
        
        **해결 방법:**
        1. GitHub 리포지토리에 `data/after/*.md` 파일이 있는지 확인
        2. Streamlit Cloud 설정에서 **Main file path**가 `app/app_after_deploy.py`인지 확인
        3. 리포지토리를 다시 clone하거나 파일 경로 확인
        """)
        st.stop()

    # MD 파일 로드
    docs = []
    load_errors = []
    
    for md_file in loader_files:
        try:
            loader = TextLoader(str(md_file), encoding='utf-8')
            file_docs = loader.load()

            for doc in file_docs:
                doc.metadata['source'] = Path(md_file).name
                doc.metadata['filepath'] = str(md_file)

            docs.extend(file_docs)
        except Exception as e:
            load_errors.append(f"{Path(md_file).name}: {str(e)}")

    if load_errors:
        with st.sidebar.expander("⚠️ 로드 실패 파일", expanded=False):
            for err in load_errors:
                st.caption(err)

    if not docs:
        st.error("❌ MD 파일을 로드했지만 내용이 비어 있습니다")
        st.stop()

    # After 설정으로 청킹
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400
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
        collection_name="after_chatbot_deploy"
    )

    return db, len(docs), len(chunks)

# 벡터 DB 초기화
with st.spinner("🔄 벡터 DB 초기화 중..."):
    try:
        db, num_docs, num_chunks = initialize_vectordb()
        st.sidebar.success(f"✅ {num_docs}개 문서, {num_chunks}개 청크")
    except Exception as e:
        st.error(f"❌ 초기화 실패: {e}")
        st.stop()

# 라우팅 키워드
RAG_KEYWORDS = [
    "환불", "불환", "반환", "돌려", "돌려받", "리펀", "refund",
    "취소", "캔슬", "cancel", "cancellation", "해지", "철회",
    "변경", "수정", "교환", "바꾸", "바꿔", "change", "modify",
    "수수료", "fee", "charge", "비용", "요금", "금액",
    "위약금", "패널티", "penalty", "벌금",
    "항공권", "티켓", "ticket", "표", "비행기표",
    "운임", "fare", "등급", "클래스",
    "flex", "flexible", "플렉스", "standard", "스탠다드",
    "saver", "세이버", "basic", "베이직",
    "특가", "할인", "일반", "프로모션",
    "국내선", "국내", "domestic", "국제선", "국제", "international",
    "노쇼", "no-show", "미탑승", "예약부도",
    "게이트", "gate", "탑승수속", "체크인", "check-in",
    "대한항공", "아시아나", "제주항공", "진에어", "티웨이", "에어서울", "이스타항공",
]

def route_to_rag(q):
    """라우팅 판단"""
    ql = q.lower()
    return any(kw.lower() in ql for kw in RAG_KEYWORDS)

def refund_rag(question, k_val, use_syn):
    """RAG 검색 및 답변"""
    search_query = expand_query_with_synonyms(question) if use_syn else question
    results = db.similarity_search_with_relevance_scores(search_query, k=k_val)

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
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state["messages"].append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        if route_to_rag(question):
            with st.spinner("🔍 검색 중..."):
                answer, sources = refund_rag(question, k, use_synonyms)

            st.markdown(answer)

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
st.caption("🟢 After 버전 - 최종 개선 (Streamlit Cloud 배포)")
st.caption("특징: chunk_size=2000, 동의어 50+, 프롬프트 개선, 대한항공 통합")
