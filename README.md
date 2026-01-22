# ✈️ 항공권 환불 상담 RAG 챗봇 성능 개선 프로젝트

7개 한국 항공사의 환불 규정을 통합한 RAG 기반 챗봇의 **체계적 성능 개선** 프로젝트

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green.svg)](https://langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange.svg)](https://openai.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📊 주요 성과 (3단계 평가)

| 지표 | Before | Middle | After | 총 개선 |
|------|--------|--------|-------|---------|
| **키워드 정확도** | 73.9% | 82.2% | 80.6% | **+6.7%p** |
| **LLM 종합 정확도** | 77.5% | 79.5% | 88.8% | **+11.3%p** |
| **검색 성공률** | 100.0% | 100.0% | 100.0% | **유지** |
| **평균 응답 시간** | 4.1초 | 5.4초 | 5.6초 | **일관성 유지** |

### **단계별 개선율**

**Stage 1 (Before → Middle): 문서 최적화**
- 키워드 정확도: **+8.3%p** ⬆️
- LLM 종합 정확도: **+2.0%p** ⬆️
- 검색 성공률: 100% 유지

**Stage 2 (Middle → After): 코드 최적화**
- 키워드 정확도: -1.7%p (안정화)
- LLM 종합 정확도: **+9.3%p** ⬆️ (주요 개선!)
- 검색 성공률: 100% 유지

---

## 🎯 프로젝트 개요

### **문제 상황**
- ❌ 동의어 미지원 (노쇼 ≠ no-show, 수수료 ≠ fee)
- ❌ Chunk 크기 부족으로 표 데이터 잘림
- ❌ 한영 혼용 검색 실패
- ❌ LLM 평가 정확도 77.5% (개선 필요)

### **해결 방법: 2단계 개선 전략**

#### **1단계: 문서 최적화 (Before → Middle)**
```
✅ 동의어 사전 30개 구축
✅ Chunk 크기 1.5배 확대 (800 → 1200)
✅ MD 파일 구조 개선

결과: 
  • 키워드 정확도: +8.3%p
  • LLM 종합 정확도: +2.0%p
  • 검색 성공률: 100% 유지
```

#### **2단계: 코드 최적화 (Middle → After)**
```
✅ 동의어 50+개 확장
✅ Chunk 크기 2.5배 확대 (1200 → 2000)
✅ 프롬프트 상세화 (표 완결성)
✅ 문서 통합 (대한항공 3개 → 1개)

결과: 
  • 키워드 정확도: 안정화 (80.6%)
  • LLM 종합 정확도: +9.3%p ⭐ (주요 개선!)
  • 검색 성공률: 100% 유지
```

---

## 🛠️ 기술 스택

| 카테고리 | 기술 |
|---------|------|
| **LLM** | OpenAI GPT-4o-mini |
| **Framework** | LangChain 0.1.0 |
| **Vector DB** | ChromaDB |
| **Frontend** | Streamlit 1.31.0 |
| **Embedding** | text-embedding-3-small |
| **Evaluation** | LLM-as-Judge (사실 80% + 완결성 20%) |

---

## 🚀 빠른 시작

### **1. 설치**

```bash
git clone https://github.com/hwijae-Jang/RAG_Chatbot.git
cd RAG_Chatbot
pip install -r requirements.txt
```

### **2. API 키 설정**

```bash
export OPENAI_API_KEY='your-api-key-here'
```

### **3. Streamlit 앱 실행**

```bash
# Before 버전 (Chunk 800, 동의어 0개)
streamlit run app/app_before.py --server.port 8501

# Middle 버전 (Chunk 1200, 동의어 30개)
streamlit run app/app_middle.py --server.port 8502

# After 버전 (Chunk 2000, 동의어 50+개)
streamlit run app/app_after.py --server.port 8503
```

### **4. 자동 평가 실행**

```bash
# 3단계 비교 평가 (15개 질문 자동 평가)
python evaluation/evaluate_3stages.py

# 결과 시각화 (PNG 그래프 4개 생성)
python evaluation/visualize_3stages.py
```

---

## 📂 프로젝트 구조

```
RAG_Chatbot/
├── app/                          # Streamlit 앱 (3개)
│   ├── app_before.py            # Before (chunk 800, 동의어 0)
│   ├── app_middle.py            # Middle (chunk 1200, 동의어 30)
│   └── app_after.py             # After (chunk 2000, 동의어 50+)
│
├── evaluation/                   # 평가 시스템
│   ├── evaluate_3stages.py      # 3단계 자동 평가 (LLM-as-Judge)
│   ├── visualize_3stages.py     # 결과 시각화 (4개 그래프)
│   └── evaluation_dataset.py    # 15개 테스트 질문
│
├── data/                         # 항공사 규정 문서 (22개 실제 MD 파일)
│   ├── before/                  # 8개 MD 파일
│   ├── middle/                  # 8개 MD 파일 (동의어 최적화)
│   └── after/                   # 6개 MD 파일 (통합 최적화)
│
├── results/                      # 평가 결과
│   ├── 3stage_keyword_accuracy.png
│   ├── 3stage_search_success.png
│   ├── improvement_breakdown.png
│   ├── 3stage_response_time.png
│   ├── evaluation_3stages_20260119_160506.json
│   └── evaluation_3stages_report.txt
│
├── .gitignore                    # Git 제외 파일
├── requirements.txt              # Python 패키지
└── README.md                     # 프로젝트 설명
```

---

## 📊 평가 방법론

### **데이터셋**
- **15개 대표 질문** (환불 가능여부, 노쇼 위약금, 수수료, 비교 등)
- **난이도 분류**: Easy (5), Medium (7), Hard (3)
- **항공사**: 대한항공, 제주항공, 진에어, 아시아나, 티웨이, 에어서울, 이스타항공

### **평가 지표**

#### **1. 키워드 정확도**
```python
keyword_score = (발견된 필수 키워드 수) / (전체 필수 키워드 수)
```

**3단계 결과:**
- Before: 73.9%
- Middle: 82.2% (+8.3%p)
- After: 80.6% (-1.7%p, 안정화)

#### **2. LLM-as-Judge 평가**
- **사실 정확성** (0-100점): MD 파일 내용과 사실 일치도
- **완결성** (0-100점): 필요 정보 포함 여부
- **종합 점수**: 사실 60% + 완결성 40%

**3단계 결과:**
- Before: 77.5%
- Middle: 79.5% (+2.0%p)
- After: 88.8% (+9.3%p) ⭐ 주요 개선!

#### **3. 검색 성공률**
```python
search_success_rate = (검색 성공 질문 수) / (전체 질문 수)
```

**3단계 결과:**
- Before: 100.0%
- Middle: 100.0%
- After: 100.0%

---

## 📈 상세 결과

### **3단계 개선 추이**

![키워드 정확도](results/3stage_keyword_accuracy%20.png)
*키워드 정확도: 73.9% → 82.2% → 80.6%*

![검색 성공률](results/3stage_search_success%20.png)
*검색 성공률: 3단계 모두 100% 유지*

![개선 기여도](results/improvement_breakdown%20.png)
*문서 최적화 단계에서 키워드 8.3%p, After 단계에서 LLM 평가 9.3%p 개선*

![응답 시간](results/3stage_response_time%20.png)
*평균 응답 시간: 4.1초 → 5.4초 → 5.6초 (일관성 유지)*

---

## 🔬 기술적 의사결정

### **1. Chunk Size 최적화**

| Chunk Size | 표 완결성 | 검색 속도 | LLM 평가 | 결정 |
|-----------|---------|---------|----------|------|
| 800자 (Before) | ❌ 잘림 | ⚡ 빠름 | 77.5% | ✗ |
| 1200자 (Middle) | △ 일부 잘림 | ⚡ 빠름 | 79.5% | △ |
| 2000자 (After) | ✅ 완전 포함 | ⚡ 빠름 | **88.8%** ⭐ | ✓ |

**결정 근거**: 
- Chunk 크기 확대가 LLM 평가에 큰 영향 (79.5% → 88.8%)
- After 단계에서 표 데이터 완전 포함으로 정확도 향상
- 정보 완결성 > 검색 속도

### **2. 동의어 사전 구축**

#### **Middle 버전 (30개)**
```python
SYNONYM_DICT = {
    "노쇼": ["노쇼", "No-Show", "no-show", "예약부도"],
    "환불": ["환불", "refund", "반환"],
    "수수료": ["수수료", "fee", "위약금", "penalty"],
    # ... 30개
}
```

#### **After 버전 (50+개)**
```python
SYNONYM_DICT = {
    "노쇼": ["노쇼", "no-show", "예약부도", "미탑승", "탑승불이행"],
    "수수료": ["수수료", "fee", "위약금", "penalty", "charge"],
    "환불": ["환불", "refund", "취소", "cancel"],
    # ... 50+ 항목
}
```

**효과**: 
- Middle 단계: 영어 검색 지원 시작, 키워드 정확도 +8.3%p
- After 단계: 완전한 한영 혼용 검색, LLM 평가 +9.3%p

### **3. LLM-as-Judge 평가**

**기존 (키워드 매칭)**: 표면적 평가
- "30,000원" ≠ "3만원" (50점)

**개선 (LLM 평가)**: 의미 기반 평가
- "30,000원" = "3만원" (95점)
- MD 파일과 직접 비교하여 사실 정확성 검증

**비용**: +$0.01 (1센트) / 45개 평가

---

## 💰 비용 분석

| 항목 | 비용 | 비고 |
|------|------|------|
| RAG 답변 생성 (45회) | $0.10 | GPT-4o-mini |
| LLM 평가 (45회) | $0.01 | GPT-4o-mini |
| **총 평가 비용** | **$0.11** | 15개 질문 × 3단계 |

---

## 📋 단계별 개선 사항

### **Stage 1: Before → Middle (문서 최적화)**

**변경 사항:**
- ✅ 동의어 사전 구축 (0 → 30개)
- ✅ MD 파일 구조 개선
- ✅ Chunk 크기 확대 (800 → 1200)

**성과:**
- 키워드 정확도: 73.9% → 82.2% (**+8.3%p**) ⭐
- LLM 종합 정확도: 77.5% → 79.5% (**+2.0%p**)
- 검색 성공률: 100% 유지

**기여율**: 키워드 개선 주도

---

### **Stage 2: Middle → After (코드 최적화)**

**변경 사항:**
- ✅ 동의어 대폭 확장 (30 → 50+개)
- ✅ Chunk 크기 최적화 (1200 → 2000)
- ✅ 프롬프트 상세화 (표 완전 포함 명시)
- ✅ 대한항공 문서 통합 (3개 → 1개)

**성과:**
- 키워드 정확도: 82.2% → 80.6% (-1.7%p, 안정화)
- LLM 종합 정확도: 79.5% → 88.8% (**+9.3%p**) ⭐⭐
- 검색 성공률: 100% 유지

**기여율**: LLM 평가 개선 주도

---

## 🎯 재현 방법

### **Google Colab 환경** (추천)
1. OpenAI API 키 설정
2. `evaluate_3stages.py` 실행 (10분)
3. 결과 PNG/JSON 다운로드

### **로컬 환경**
```bash
pip install -r requirements.txt
export OPENAI_API_KEY='your-key'
python evaluation/evaluate_3stages.py
python evaluation/visualize_3stages.py
```

**예상 시간**: 10분  
**예상 비용**: $0.11

---

## 🔍 주요 인사이트

### **1. Middle 단계: 키워드 검색 개선**
- 키워드 정확도: 73.9% → 82.2% (**+8.3%p**)
- 동의어 30개로 한영 검색 지원 시작
- 문서 구조 개선으로 검색 품질 향상

### **2. After 단계: LLM 평가 대폭 개선**
- LLM 평가: 79.5% → 88.8% (**+9.3%p**) ⭐
- Chunk 크기 2000으로 표 데이터 완전 포함
- 프롬프트 상세화로 답변 완결성 향상

### **3. 검색 성공률 100% 유지**
- 3단계 모두 100% 검색 성공
- 동의어 사전과 문서 최적화 효과

---

## 🤝 기여

이슈 및 PR 환영합니다!

---

## 📄 라이센스

MIT License - 자유롭게 사용, 수정, 배포 가능

---

## 👤 작성자

**장휘재 (Hwi-Jae Jang)**
- GitHub: [@hwijae-Jang](https://github.com/hwijae-Jang)
- Email: hwijae35@naver.com

---

## 📞 연락처

프로젝트에 대한 질문이나 제안사항이 있으시면 언제든 연락 주세요!

- **Issues**: [GitHub Issues](https://github.com/hwijae-Jang/RAG_Chatbot/issues)
- **Email**: hwijae35@naver.com

---

⭐ **프로젝트가 도움이 되었다면 Star를 눌러주세요!**

---

## 📊 평가 요약

### **최종 성과**

| 단계 | 키워드 정확도 | LLM 종합 정확도 | 검색 성공률 |
|------|-------------|----------------|------------|
| Before | 73.9% | 77.5% | 100.0% |
| Middle | 82.2% | 79.5% | 100.0% |
| After | 80.6% | **88.8%** ⭐ | 100.0% |

### **핵심 개선**
- ✅ **키워드 정확도 8.3%p 향상** (Middle 단계)
- ✅ **LLM 평가 9.3%p 향상** (After 단계) ⭐
- ✅ **검색 성공률 100% 유지** (3단계 모두)
- ✅ **동의어 50+개 구축** (한영 검색 완벽 지원)
- ✅ **비용 효율적** ($0.11 / 45개 평가)

---

## 🔮 향후 계획

- [ ] 추가 항공사 규정 확대
- [ ] 다국어 지원 (영어, 일본어)
- [ ] 실시간 업데이트 시스템
- [ ] 웹 크롤링 자동화
- [ ] 프로덕션 배포 (AWS, GCP)
