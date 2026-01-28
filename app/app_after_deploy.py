
"""
After 버전 항공권 환불 상담 RAG 챗봇
- chunk_size: 2000
- chunk_overlap: 400
- 동의어 사전: 50+
- 대한항공 통합 (6개 md 파일)
"""

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
st.set_page_config(page_title="After - 항공권 환불 챗봇", layout="wide")
st.title("✈️ After 버전 - 환불 상담 챗봇")
st.markdown("### 🟢 최종 개선 버전")
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
            # 동의어 추가
            synonyms = SYNONYM_DICT[word_lower]
            expanded_terms.extend(synonyms[:3])  # 최대 3개
        expanded_terms.append(word)

    return " ".join(expanded_terms)

# 사이드바
with st.sidebar:
    st.header("📚 After 설정")
    st.success("✅ 최종 개선 버전")

    # API 키 입력
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
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

# 프롬프트 (더 상세하게 개선)
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
    """After 설정으로 벡터 DB 초기화"""

    # MD 파일 경로
    # MD 파일 경로 패턴 (Streamlit Cloud 배포 최적화)
    import glob
    patterns = [
        "data/after/*.md",       # Streamlit Cloud (상대 경로)
        "./data/after/*.md",     # 로컬 실행
        "../data/after/*.md",    # app/ 폴더에서 실행 시
    ]

    # 파일 검색
    seen = set()
    md_files = []

    for pat in patterns:
        for fp in glob.glob(pat, recursive=True):
            if fp.endswith(".md") and fp not in seen and Path(fp).is_file():
                seen.add(fp)
                md_files.append(fp)

    if not md_files:
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
        """)
        st.stop()


    # MD 파일 로드
    docs = []
    # MD 파일 로드
    for md_file in md_files:
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

    # After 설정으로 청킹 (큰 청크로 표 완전 포함)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # After: 큰 청크 (표 잘림 방지)
        chunk_overlap=400  # 오버랩 증가
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
        collection_name="after_chatbot"
    )

    return db, len(docs), len(chunks)

with st.spinner("🔄 벡터 DB 초기화 중..."):
    db, num_docs, num_chunks = initialize_vectordb()

st.sidebar.success(f"✅ {num_docs}개 문서, {num_chunks}개 청크")

# 라우팅 키워드 (After 버전 - 대폭 확장)
RAG_KEYWORDS = [
    # === 환불/취소 관련 ===
    "환불", "불환", "반환", "돌려", "돌려받", "리펀", "refund",
    "취소", "캔슬", "cancel", "cancellation", "해지", "철회",
    "부분환불", "전액환불", "일부환불",

    # === 변경 관련 ===
    "변경", "수정", "교환", "바꾸", "바꿔", "change", "modify", "modification",
    "일정변경", "날짜변경", "시간변경", "편명변경", "경로변경",
    "재발권", "리이슈", "reissue",

    # === 수수료 관련 ===
    "수수료", "fee", "charge", "비용", "요금", "가격", "금액",
    "위약금", "패널티", "penalty", "벌금",
    "변경수수료", "환불수수료", "취소수수료", "재발권수수료",
    "무료", "공짜", "꽁짜", "꽁자", "꽁자", "free", "면제",

    # === 항공권/티켓 관련 ===
    "항공권", "티켓", "ticket", "표", "비행기표", "항공", "항공편",
    "편명", "좌석", "seat", "booking", "예약",

    # === 운임 등급 ===
    "운임", "fare", "등급", "클래스", "class",

    # 기본 운임 등급 (제주항공, 대한항공 등)
    "flex", "flexible", "플렉스", "플렉시블",
    "standard", "스탠다드",
    "saver", "세이버", "save",
    "basic", "베이직", "베이식",

    # 이스타항공/아시아나 운임 종류
    "특가", "특가운임", "프로모션", "promotion", "special",
    "할인", "할인운임", "discount", "세일",
    "일반", "일반운임", "정상", "정상운임", "regular", "normal",

    # 좌석 등급
    "premium", "프리미엄", "비즈", "biz", "business","비즈니스",
    "biz lite", "비즈라이트", "flex", "플렉스","플랙스", "standard", "스탠다드","스텐다드",
    "saver", "세이버","새이버", "basic", "베이식","베이직","배이식","배이직",
    "이코노미", "economy", "일반석", "비즈니스석", "비즈니스","일등석", "퍼스트","first",
    "프리미엄 이코노미", "프레스티지","prestige",
    "W","T","L","S","V","K","H","E","Q","M","B","Y","Z","U","D","C","J",
    "지니플러스","지니","플렉스","플랙스","슈퍼로우","지니비즈"

    # === 노선 구분 ===
    "국내선", "국내", "domestic", "도메스틱",
    "국제선", "국제", "international", "인터내셔널", "해외", "외국"
    "단거리", "중거리", "장거리", "short", "medium", "long",

    # === 노쇼 관련 ===
    "노쇼", "no-show", "noshow", "미탑승", "미승선", "불탑승",
    "미취소", "미출현", "불출석", "예약부도",
    "게이트", "gate", "출구장", "탑승구",
    "탑승수속", "체크인", "check-in", "수속",

    # === 기간/시간 관련 ===
    "기간", "기한", "유효", "유효기간", "validity", "만료",
    "출발", "출발일", "출발전", "출발후", "departure",
    "당일", "오늘", "며칠", "몇일", "며칠전", "일전", "전",
    "이전", "이후", "before", "after",
    "91일", "90일", "60일", "15일", "14일", "4일", "3일",

    # === 규정/정책 관련 ===
    "규정", "정책", "policy", "약관", "조건", "규칙", "rule",
    "가능", "불가", "가능한", "안되", "되나", "할수있", "할수없",

    # === 항공사명 ===
    "대한항공", "아시아나", "제주항공", "진에어", "티웨이","에어서울","이스타항공",
    "korean", "koreanair", "asiana", "jeju", "jejuair", "jin", "jinair", "tway","airseoul","eastar"

    # === 질문 키워드 ===
    "언제", "when", "얼마", "how much", "어디", "where",
    "무엇", "what", "왜", "why",
    "가능해", "되나요", "인가요", "한가요", "나요"
]

def route_to_rag(q):
    """라우팅 판단"""
    ql = q.lower()
    return any(kw.lower() in ql for kw in RAG_KEYWORDS)

def refund_rag(question, k_val, use_syn):
    """RAG 검색 및 답변 (동의어 확장 옵션)"""

    # 동의어 확장
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
    # 사용자 메시지
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state["messages"].append({"role": "user", "content": question})

    # 어시스턴트 답변
    with st.chat_message("assistant"):
        if route_to_rag(question):
            with st.spinner("🔍 검색 중..."):
                answer, sources = refund_rag(question, k, use_synonyms)

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
st.caption("🟢 After 버전 - 최종 개선")
st.caption("특징: chunk_size=2000, 동의어 50+, 프롬프트 개선, 대한항공 통합")
