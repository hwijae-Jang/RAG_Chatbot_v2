"""
평가용 테스트 데이터셋
각 질문에 대한 정답과 평가 기준 포함
"""

EVALUATION_DATASET = [
    {
        "id": 1,
        "question": "제주항공 국제선 FLEX 운임은 출발 전 환불이 가능한가요?",
        "expected_answer": "가능",
        "keywords_required": ["FLEX", "무료", "환불", "가능"],
        "airline": "제주항공",
        "category": "환불_가능여부",
        "difficulty": "easy"
    },
    {
        "id": 2,
        "question": "진에어 국내선 노쇼 위약금은 얼마인가요?",
        "expected_answer": "편도 30,000원",
        "keywords_required": ["30,000", "편도", "국내선"],
        "airline": "진에어",
        "category": "노쇼_위약금",
        "difficulty": "easy"
    },
    {
        "id": 3,
        "question": "대한항공 국제선 출발 5일 전 변경하면 수수료가 얼마인가요?",
        "expected_answer": "운임에 따라 다름",
        "keywords_required": ["수수료", "운임", "등급"],
        "airline": "대한항공",
        "category": "변경_수수료",
        "difficulty": "medium"
    },
    {
        "id": 4,
        "question": "제주항공 BASIC 운임과 FLEX 운임의 환불 규정 차이는?",
        "expected_answer": "BASIC은 환불 불가, FLEX는 무료 환불",
        "keywords_required": ["BASIC", "불가", "FLEX", "무료"],
        "airline": "제주항공",
        "category": "운임_비교",
        "difficulty": "hard"
    },
    {
        "id": 5,
        "question": "아시아나 특가 운임은 환불이 되나요?",
        "expected_answer": "환불 불가 또는 제한적",
        "keywords_required": ["특가", "환불"],
        "airline": "아시아나",
        "category": "환불_가능여부",
        "difficulty": "easy"
    },
    {
        "id": 6,
        "question": "티웨이 국제선 출발 3일 전 취소 시 수수료는?",
        "expected_answer": "운임에 따라 수수료 발생",
        "keywords_required": ["수수료", "3일", "취소"],
        "airline": "티웨이",
        "category": "환불_수수료",
        "difficulty": "medium"
    },
    {
        "id": 7,
        "question": "에어서울 체크인 후 탑승하지 않으면 어떻게 되나요?",
        "expected_answer": "노쇼 위약금 부과",
        "keywords_required": ["노쇼", "위약금", "탑승"],
        "airline": "에어서울",
        "category": "노쇼_위약금",
        "difficulty": "medium"
    },
    {
        "id": 8,
        "question": "이스타항공 일반운임 국내선 환불 수수료는?",
        "expected_answer": "출발 시점에 따라 다름",
        "keywords_required": ["일반", "환불", "수수료"],
        "airline": "이스타항공",
        "category": "환불_수수료",
        "difficulty": "medium"
    },
    {
        "id": 9,
        "question": "진에어와 제주항공의 노쇼 위약금을 비교해주세요.",
        "expected_answer": "두 항공사 모두 노쇼 위약금 있음",
        "keywords_required": ["진에어", "제주항공", "노쇼"],
        "airline": "복수",
        "category": "항공사_비교",
        "difficulty": "hard"
    },
    {
        "id": 10,
        "question": "대한항공 프리미엄 이코노미 환불 규정은?",
        "expected_answer": "운임 등급에 따른 환불 규정 적용",
        "keywords_required": ["환불", "운임"],
        "airline": "대한항공",
        "category": "환불_가능여부",
        "difficulty": "medium"
    },
    # 추가 질문들 (총 30개 예시)
    {
        "id": 11,
        "question": "제주항공 국제선 STANDARD 운임 출발 15일 전 환불 수수료는?",
        "expected_answer": "10,000원",
        "keywords_required": ["STANDARD", "10,000", "수수료"],
        "airline": "제주항공",
        "category": "환불_수수료",
        "difficulty": "easy"
    },
    {
        "id": 12,
        "question": "노쇼란 무엇인가요?",
        "expected_answer": "탑승수속 후 탑승하지 않는 것",
        "keywords_required": ["노쇼", "탑승", "미탑승"],
        "airline": "일반",
        "category": "용어_설명",
        "difficulty": "easy"
    },
    {
        "id": 13,
        "question": "진에어 국제선 노쇼 위약금은 왕복 기준 얼마인가요?",
        "expected_answer": "100,000원",
        "keywords_required": ["100,000", "왕복", "국제선"],
        "airline": "진에어",
        "category": "노쇼_위약금",
        "difficulty": "easy"
    },
    {
        "id": 14,
        "question": "항공권을 환불받으려면 어떻게 해야 하나요?",
        "expected_answer": "항공사 또는 구매처에 환불 신청",
        "keywords_required": ["환불", "신청"],
        "airline": "일반",
        "category": "절차_안내",
        "difficulty": "easy"
    },
    {
        "id": 15,
        "question": "대한항공 국내선과 국제선의 환불 규정 차이는?",
        "expected_answer": "국내선과 국제선 수수료 차이 있음",
        "keywords_required": ["국내선", "국제선", "차이"],
        "airline": "대한항공",
        "category": "노선_비교",
        "difficulty": "hard"
    },
]

# 평가 기준
EVALUATION_CRITERIA = {
    "정확도": {
        "완전정답": 1.0,      # 정답과 완전히 일치
        "부분정답": 0.7,      # 핵심 내용은 맞지만 세부사항 다름
        "관련답변": 0.4,      # 관련은 있지만 정확하지 않음
        "오답": 0.0           # 완전히 틀림
    },
    "키워드_포함": {
        "모두포함": 1.0,      # 필수 키워드 모두 포함
        "대부분포함": 0.7,    # 70% 이상 포함
        "일부포함": 0.4,      # 40% 이상 포함
        "미포함": 0.0         # 40% 미만
    },
    "근거_제시": {
        "명확함": 1.0,        # 출처와 유사도 점수 제시
        "일부제시": 0.5,      # 출처만 제시
        "없음": 0.0           # 근거 없음
    }
}
