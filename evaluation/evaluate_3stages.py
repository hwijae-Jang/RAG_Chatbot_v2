
"""
GPT 성능비교 가중치 조절
Before → Middle → After 3단계 비교 평가 스크립트
- 15개 테스트 질문으로 체계적 비교
- 각 단계별 개선 효과 정량화
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# 테스트 데이터셋
TEST_DATASET = [
    {
        "id": 1,
        "question": "제주항공 국제선 FLEX 운임은 출발 전 환불이 가능한가요?",
        "keywords": ["FLEX", "무료", "환불", "가능"],
        "airline": "제주항공",
        "category": "환불_가능여부"
    },
    {
        "id": 2,
        "question": "진에어 국내선 노쇼 위약금은 얼마인가요?",
        "keywords": ["30,000", "편도", "국내선"],
        "airline": "진에어",
        "category": "노쇼_위약금"
    },
    {
        "id": 3,
        "question": "대한항공 국제선 출발 5일 전 변경하면 수수료가 얼마인가요?",
        "keywords": ["수수료", "운임", "등급"],
        "airline": "대한항공",
        "category": "변경_수수료"
    },
    {
        "id": 4,
        "question": "제주항공 BASIC 운임과 FLEX 운임의 환불 규정 차이는?",
        "keywords": ["BASIC", "불가", "FLEX", "무료"],
        "airline": "제주항공",
        "category": "운임_비교"
    },
    {
        "id": 5,
        "question": "아시아나 특가 운임은 환불이 되나요?",
        "keywords": ["특가", "환불"],
        "airline": "아시아나",
        "category": "환불_가능여부"
    },
    {
        "id": 6,
        "question": "티웨이 국제선 출발 3일 전 취소 시 수수료는?",
        "keywords": ["수수료", "3일", "취소"],
        "airline": "티웨이",
        "category": "환불_수수료"
    },
    {
        "id": 7,
        "question": "에어서울 체크인 후 탑승하지 않으면 어떻게 되나요?",
        "keywords": ["노쇼", "위약금", "탑승"],
        "airline": "에어서울",
        "category": "노쇼_위약금"
    },
    {
        "id": 8,
        "question": "이스타항공 일반운임 국내선 환불 수수료는?",
        "keywords": ["일반", "환불", "수수료"],
        "airline": "이스타항공",
        "category": "환불_수수료"
    },
    {
        "id": 9,
        "question": "진에어와 제주항공의 노쇼 위약금을 비교해주세요.",
        "keywords": ["진에어", "제주항공", "노쇼"],
        "airline": "복수",
        "category": "항공사_비교"
    },
    {
        "id": 10,
        "question": "대한항공 프리미엄 이코노미 환불 규정은?",
        "keywords": ["환불", "운임"],
        "airline": "대한항공",
        "category": "환불_가능여부"
    },
    {
        "id": 11,
        "question": "제주항공 국제선 STANDARD 운임 출발 15일 전 환불 수수료는?",
        "keywords": ["STANDARD", "수수료"],
        "airline": "제주항공",
        "category": "환불_수수료"
    },
    {
        "id": 12,
        "question": "노쇼란 무엇인가요?",
        "keywords": ["노쇼", "탑승", "미탑승"],
        "airline": "일반",
        "category": "용어_설명"
    },
    {
        "id": 13,
        "question": "진에어 국제선 노쇼 위약금은 왕복 기준 얼마인가요?",
        "keywords": ["100,000", "왕복", "국제선"],
        "airline": "진에어",
        "category": "노쇼_위약금"
    },
    {
        "id": 14,
        "question": "항공권을 환불받으려면 어떻게 해야 하나요?",
        "keywords": ["환불", "신청"],
        "airline": "일반",
        "category": "절차_안내"
    },
    {
        "id": 15,
        "question": "대한항공 국내선과 국제선의 환불 규정 차이는?",
        "keywords": ["국내선", "국제선", "차이"],
        "airline": "대한항공",
        "category": "노선_비교"
    }
]


class RAGEvaluator:
    """RAG 시스템 평가기"""

    def __init__(self, version: str, md_path: str, chunk_size: int, chunk_overlap: int,
                 synonyms: dict = None):
        self.version = version
        self.md_path = md_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.synonyms = synonyms or {}

        # API 키 확인
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다")

        # LLM 초기화
        self.llm = ChatOpenAI(
            model='gpt-4o-mini',
            temperature=0,
            api_key=self.api_key
        )

        # 프롬프트
        self.rag_chain = (
            ChatPromptTemplate.from_template("""
너는 항공권 환불 및 변경을 도와주는 친절한 한국어 상담 챗봇이야.
아래 항공사 정책 문서를 참고해서 질문에 정확하고 친절하게 답변해줘.

정책문서: {context}
사용자 질문: {q}

답변:
""") | self.llm | StrOutputParser()
        )

        self.db = None

    def initialize(self):
        """벡터 DB 초기화"""
        print(f"\n🔧 {self.version} 버전 초기화...")
        print(f"   Chunk Size: {self.chunk_size}")
        print(f"   Chunk Overlap: {self.chunk_overlap}")
        print(f"   Synonyms: {len(self.synonyms)}개")

        # MD 파일 로드
        docs = []
        md_dir = Path(self.md_path)

        for md_file in md_dir.glob("*.md"):
            try:
                loader = TextLoader(str(md_file), encoding='utf-8')
                file_docs = loader.load()

                for doc in file_docs:
                    doc.metadata['source'] = md_file.name

                docs.extend(file_docs)
            except Exception as e:
                print(f"   ⚠️ 파일 로드 실패: {md_file.name}")
                continue

        print(f"   📁 {len(docs)}개 문서 로드")

        # 청킹
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        print(f"   📄 {len(chunks)}개 청크 생성")

        # 벡터 DB
        embeddings = OpenAIEmbeddings(
            model='text-embedding-3-small',
            api_key=self.api_key
        )

        self.db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=f"{self.version}_eval"
        )

        print(f"   ✅ 초기화 완료")

    def expand_query(self, query: str) -> str:
        """동의어로 쿼리 확장"""
        if not self.synonyms:
            return query

        words = query.split()
        expanded = []

        for word in words:
            word_lower = word.lower()
            if word_lower in self.synonyms:
                expanded.extend(self.synonyms[word_lower][:2])
            expanded.append(word)

        return " ".join(expanded)

    def evaluate_answer_accuracy_with_llm(self, question: str, answer: str, ground_truth_context: str) -> Dict:
        """
        LLM-as-Judge: GPT로 답변 정확도 평가

        Args:
            question: 사용자 질문
            answer: RAG 시스템의 답변
            ground_truth_context: MD 파일에서 검색된 실제 내용 (정답 근거)

        Returns:
            {
                "accuracy_score": float (0-1),
                "factual_correctness": float (0-1),
                "completeness": float (0-1),
                "evaluation_reason": str
            }
        """

        evaluation_prompt = f"""당신은 항공권 환불 규정 전문가입니다.
아래 질문에 대한 챗봇 답변이 정답 근거 자료와 비교했을 때 얼마나 정확한지 평가해주세요.

**질문**: {question}

**정답 근거 (MD 파일 내용)**:
{ground_truth_context}

**챗봇 답변**:
{answer}

다음 2가지 기준으로 0-100점 척도로 평가하고, JSON 형식으로 답변하세요:

1. **사실 정확성 (factual_correctness)**: 챗봇 답변이 정답 근거와 사실적으로 일치하는가? (0-100)
2. **완결성 (completeness)**: 질문에 필요한 정보를 충분히 포함하고 있는가? (0-100)

**중요**:
- 챗봇 답변이 정답 근거에 없는 내용을 지어냈다면 factual_correctness를 0점으로 평가
- 정답 근거에 있는 핵심 정보를 누락했다면 completeness를 낮게 평가
- 표현이 다르더라도 의미가 같으면 높은 점수 부여

다음 JSON 형식으로만 답변하세요 (다른 설명 없이):
{{
    "factual_correctness": 85,
    "completeness": 90,
    "evaluation_reason": "평가 이유를 1-2문장으로"
}}
"""

        try:
            # LLM으로 평가
            eval_llm = ChatOpenAI(
                model='gpt-4o-mini',
                temperature=0,
                api_key=self.api_key
            )

            eval_result = eval_llm.invoke(evaluation_prompt)

            # JSON 파싱
            import json
            import re

            # JSON 부분만 추출
            content = eval_result.content
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)

            if json_match:
                eval_data = json.loads(json_match.group())

                # GPT 응답에서 점수 추출
                factual = eval_data.get("factual_correctness", 0) / 100
                complete = eval_data.get("completeness", 0) / 100

                # ========================================
                # 🎯 가중치 조정
                # ========================================
                # 옵션 1: 사실 중시 (80% + 20%)
                accuracy = (factual * 0.8) + (complete * 0.2)

                # 옵션 2: 균형 (60% + 40%) - 기본값
                # accuracy = (factual * 0.6) + (complete * 0.4)

                # 옵션 3: 완결성 중시 (40% + 60%)
                # accuracy = (factual * 0.4) + (complete * 0.6)
                # ========================================

                return {
                    "accuracy_score": accuracy,  # 재계산된 종합 점수
                    "factual_correctness": factual,
                    "completeness": complete,
                    "evaluation_reason": eval_data.get("evaluation_reason", "평가 실패")
                }
            else:
                return {
                    "accuracy_score": 0.0,
                    "factual_correctness": 0.0,
                    "completeness": 0.0,
                    "evaluation_reason": "JSON 파싱 실패"
                }

        except Exception as e:
            print(f"   ⚠️ LLM 평가 실패: {e}")
            return {
                "accuracy_score": 0.0,
                "factual_correctness": 0.0,
                "completeness": 0.0,
                "evaluation_reason": f"오류: {str(e)}"
            }

    def evaluate_single(self, test: Dict) -> Dict:
        """단일 질문 평가"""
        question = test["question"]
        keywords = test["keywords"]

        # 검색
        start_time = time.time()
        search_query = self.expand_query(question)
        results = self.db.similarity_search_with_relevance_scores(search_query, k=3)
        search_time = time.time() - start_time

        # 답변 생성
        if results:
            context = "\n\n".join([doc.page_content for doc, _ in results])
            gen_start = time.time()
            answer = self.rag_chain.invoke({'context': context, 'q': question})
            gen_time = time.time() - gen_start
        else:
            context = ""
            answer = "관련 규정을 찾지 못했습니다."
            gen_time = 0

        total_time = time.time() - start_time

        # 키워드 매칭 (기존 방식)
        answer_lower = answer.lower()
        keywords_found = [kw for kw in keywords if kw.lower() in answer_lower]
        keyword_score = len(keywords_found) / len(keywords) if keywords else 0

        # LLM-as-Judge 평가 (새로운 방식)
        if context:
            llm_evaluation = self.evaluate_answer_accuracy_with_llm(question, answer, context)
        else:
            llm_evaluation = {
                "accuracy_score": 0.0,
                "factual_correctness": 0.0,
                "completeness": 0.0,
                "evaluation_reason": "검색 결과 없음"
            }

        return {
            "id": test["id"],
            "question": question,
            "answer": answer,
            "keywords_expected": keywords,
            "keywords_found": keywords_found,
            "keyword_score": keyword_score,
            "search_success": len(results) > 0,
            "num_sources": len(results),
            "search_time": search_time,
            "generation_time": gen_time,
            "total_time": total_time,
            "category": test["category"],
            # LLM 평가 추가
            "llm_accuracy_score": llm_evaluation["accuracy_score"],
            "llm_factual_correctness": llm_evaluation["factual_correctness"],
            "llm_completeness": llm_evaluation["completeness"],
            "llm_evaluation_reason": llm_evaluation["evaluation_reason"]
        }

    def evaluate_all(self, dataset: List[Dict]) -> List[Dict]:
        """전체 데이터셋 평가"""
        print(f"\n📊 {self.version} 평가 시작 ({len(dataset)}개 질문)")

        results = []
        for i, test in enumerate(dataset, 1):
            print(f"[{i}/{len(dataset)}] {test['question'][:40]}...")

            result = self.evaluate_single(test)

            print(f"   키워드: {result['keyword_score']:.0%} | LLM평가: {result['llm_accuracy_score']:.0%} | 검색: {'✅' if result['search_success'] else '❌'} | 시간: {result['total_time']:.2f}s")
            if result['llm_evaluation_reason']:
                print(f"   📝 {result['llm_evaluation_reason']}")

            results.append(result)

        return results


def calculate_metrics(results: List[Dict]) -> Dict:
    """종합 지표 계산"""

    keyword_scores = [r['keyword_score'] for r in results]
    avg_keyword = sum(keyword_scores) / len(keyword_scores)

    # LLM 평가 점수 추가
    llm_accuracy_scores = [r.get('llm_accuracy_score', 0) for r in results]
    llm_factual_scores = [r.get('llm_factual_correctness', 0) for r in results]
    llm_completeness_scores = [r.get('llm_completeness', 0) for r in results]

    avg_llm_accuracy = sum(llm_accuracy_scores) / len(llm_accuracy_scores)
    avg_llm_factual = sum(llm_factual_scores) / len(llm_factual_scores)
    avg_llm_completeness = sum(llm_completeness_scores) / len(llm_completeness_scores)

    search_successes = [1 for r in results if r['search_success']]
    search_rate = len(search_successes) / len(results)

    times = [r['total_time'] for r in results]
    avg_time = sum(times) / len(times)

    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r['keyword_score'])

    category_scores = {
        cat: sum(scores) / len(scores)
        for cat, scores in categories.items()
    }

    return {
        "avg_keyword_score": avg_keyword,
        "avg_llm_accuracy": avg_llm_accuracy,
        "avg_llm_factual": avg_llm_factual,
        "avg_llm_completeness": avg_llm_completeness,
        "search_success_rate": search_rate,
        "avg_response_time": avg_time,
        "category_scores": category_scores
    }


def main():
    """메인 실행"""
    print("="*70)
    print("🚀 Before → Middle → After 3단계 비교 평가")
    print("="*70)

    # Before 동의어 사전 (없음)
    BEFORE_SYNONYMS = {}

    # Middle 동의어 사전 (30개)
    MIDDLE_SYNONYMS = {
        "노쇼": ["노쇼", "no-show", "예약부도"],
        "no-show": ["노쇼", "no-show", "예약부도"],
        "예약부도": ["노쇼", "예약부도"],
        "환불": ["환불", "refund"],
        "refund": ["환불", "refund"],
        "수수료": ["수수료", "fee", "penalty"],
        "fee": ["수수료", "fee"],
        "위약금": ["위약금", "penalty"],
        "flex": ["플렉스", "FLEX"],
        "basic": ["베이직", "BASIC"],
    }

    # After 동의어 사전 (50+)
    AFTER_SYNONYMS = {
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

    # Before 평가
    before_eval = RAGEvaluator(
        version="Before",
        md_path="/content/before",
        chunk_size=800,
        chunk_overlap=100,
        synonyms=BEFORE_SYNONYMS
    )
    before_eval.initialize()
    before_results = before_eval.evaluate_all(TEST_DATASET)
    before_metrics = calculate_metrics(before_results)

    # Middle 평가
    middle_eval = RAGEvaluator(
        version="Middle",
        md_path="/content/middle",
        chunk_size=1200,
        chunk_overlap=200,
        synonyms=MIDDLE_SYNONYMS
    )
    middle_eval.initialize()
    middle_results = middle_eval.evaluate_all(TEST_DATASET)
    middle_metrics = calculate_metrics(middle_results)

    # After 평가
    after_eval = RAGEvaluator(
        version="After",
        md_path="/content/after",
        chunk_size=2000,
        chunk_overlap=400,
        synonyms=AFTER_SYNONYMS
    )
    after_eval.initialize()
    after_results = after_eval.evaluate_all(TEST_DATASET)
    after_metrics = calculate_metrics(after_results)

    # 결과 출력
    print("\n" + "="*70)
    print("📊 3단계 평가 결과 요약")
    print("="*70)

    print(f"\n{'지표':<25} {'Before':>15} {'Middle':>15} {'After':>15}")
    print("-"*70)

    # 키워드 정확도
    print(f"{'키워드 정확도':<25} {before_metrics['avg_keyword_score']:>14.1%} {middle_metrics['avg_keyword_score']:>14.1%} {after_metrics['avg_keyword_score']:>14.1%}")

    # LLM 평가 점수 추가
    print(f"{'LLM 종합 정확도':<25} {before_metrics['avg_llm_accuracy']:>14.1%} {middle_metrics['avg_llm_accuracy']:>14.1%} {after_metrics['avg_llm_accuracy']:>14.1%}")
    print(f"{'LLM 사실 정확성':<25} {before_metrics['avg_llm_factual']:>14.1%} {middle_metrics['avg_llm_factual']:>14.1%} {after_metrics['avg_llm_factual']:>14.1%}")
    print(f"{'LLM 완결성':<25} {before_metrics['avg_llm_completeness']:>14.1%} {middle_metrics['avg_llm_completeness']:>14.1%} {after_metrics['avg_llm_completeness']:>14.1%}")

    # 검색 성공률
    print(f"{'검색 성공률':<25} {before_metrics['search_success_rate']:>14.1%} {middle_metrics['search_success_rate']:>14.1%} {after_metrics['search_success_rate']:>14.1%}")

    # 응답 시간
    print(f"{'평균 응답 시간':<25} {before_metrics['avg_response_time']:>13.2f}s {middle_metrics['avg_response_time']:>13.2f}s {after_metrics['avg_response_time']:>13.2f}s")

    # 개선율 계산
    print("\n" + "="*70)
    print("📈 단계별 개선율")
    print("="*70)

    kw_before_middle = (middle_metrics['avg_keyword_score'] - before_metrics['avg_keyword_score']) * 100
    kw_middle_after = (after_metrics['avg_keyword_score'] - middle_metrics['avg_keyword_score']) * 100
    kw_before_after = (after_metrics['avg_keyword_score'] - before_metrics['avg_keyword_score']) * 100

    print(f"\n키워드 정확도:")
    print(f"   Before → Middle: +{kw_before_middle:.1f}%p")
    print(f"   Middle → After:  +{kw_middle_after:.1f}%p")
    print(f"   Before → After:  +{kw_before_after:.1f}%p (총 개선)")

    # LLM 평가 개선율
    llm_before_middle = (middle_metrics['avg_llm_accuracy'] - before_metrics['avg_llm_accuracy']) * 100
    llm_middle_after = (after_metrics['avg_llm_accuracy'] - middle_metrics['avg_llm_accuracy']) * 100
    llm_before_after = (after_metrics['avg_llm_accuracy'] - before_metrics['avg_llm_accuracy']) * 100

    print(f"\nLLM 종합 정확도:")
    print(f"   Before → Middle: +{llm_before_middle:.1f}%p")
    print(f"   Middle → After:  +{llm_middle_after:.1f}%p")
    print(f"   Before → After:  +{llm_before_after:.1f}%p (총 개선)")

    sr_before_middle = (middle_metrics['search_success_rate'] - before_metrics['search_success_rate']) * 100
    sr_middle_after = (after_metrics['search_success_rate'] - middle_metrics['search_success_rate']) * 100
    sr_before_after = (after_metrics['search_success_rate'] - before_metrics['search_success_rate']) * 100

    print(f"\n검색 성공률:")
    print(f"   Before → Middle: +{sr_before_middle:.1f}%p")
    print(f"   Middle → After:  +{sr_middle_after:.1f}%p")
    print(f"   Before → After:  +{sr_before_after:.1f}%p (총 개선)")

    # 결과 저장
    output = {
        "timestamp": datetime.now().isoformat(),
        "before": {
            "results": before_results,
            "metrics": before_metrics
        },
        "middle": {
            "results": middle_results,
            "metrics": middle_metrics
        },
        "after": {
            "results": after_results,
            "metrics": after_metrics
        }
    }

    output_file = f"evaluation_3stages_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n📄 결과 저장: {output_file}")

    return output


if __name__ == "__main__":
    main()

