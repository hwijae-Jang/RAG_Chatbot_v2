# 🚀 빠른 시작 가이드

RAG_Chatbot 프로젝트를 5분 안에 실행하는 방법

---

## ⚡ 빠른 실행 (Google Colab)

### **1. Repository Clone**

```python
!git clone https://github.com/hwijae-Jang/RAG_Chatbot.git
%cd RAG_Chatbot
```

### **2. 패키지 설치**

```python
!pip install -q langchain langchain-openai langchain-community chromadb streamlit matplotlib openai
```

### **3. API 키 설정**

```python
import os
from google.colab import userdata

# Colab Secrets에 저장된 키 사용
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')

# 또는 직접 입력
# os.environ["OPENAI_API_KEY"] = "sk-..."
```

### **4. Streamlit 앱 실행**

```python
# Colab에서 Streamlit 실행
!streamlit run app/app_after.py &>/dev/null&

# LocalTunnel로 외부 접속
!npx localtunnel --port 8501
```

### **5. 평가 실행**

```python
!python evaluation/evaluate_3stages.py
!python evaluation/visualize_3stages.py

# 결과 확인
from IPython.display import Image, display
display(Image('results/3stage_keyword_accuracy.png'))
```

---

## 💻 로컬 실행

### **1. Clone & Install**

```bash
git clone https://github.com/hwijae-Jang/RAG_Chatbot.git
cd RAG_Chatbot
pip install -r requirements.txt
```

### **2. API 키 설정**

**macOS/Linux:**
```bash
export OPENAI_API_KEY='sk-your-key-here'
```

**Windows:**
```cmd
set OPENAI_API_KEY=sk-your-key-here
```

### **3. 실행**

```bash
# Before 버전
streamlit run app/app_before.py --server.port 8501

# Middle 버전
streamlit run app/app_middle.py --server.port 8502

# After 버전 (최적화)
streamlit run app/app_after.py --server.port 8503
```

브라우저에서 접속:
- http://localhost:8501 (Before)
- http://localhost:8502 (Middle)
- http://localhost:8503 (After)

---

## 🧪 평가 실행

```bash
# 3단계 자동 평가 (10분 소요)
python evaluation/evaluate_3stages.py

# 결과 시각화 (PNG 4개 생성)
python evaluation/visualize_3stages.py
```

---

## 🐛 문제 해결

### **문제 1: OpenAI API 키 오류**

```python
# 키 확인
import os
print(os.environ.get("OPENAI_API_KEY", "키 없음"))

# 키 재설정
os.environ["OPENAI_API_KEY"] = "sk-..."
```

### **문제 2: ChromaDB 에러**

```bash
# ChromaDB 재설치
pip uninstall chromadb -y
pip install chromadb==0.4.22
```

### **문제 3: Streamlit 포트 충돌**

```bash
# 다른 포트 사용
streamlit run app/app_after.py --server.port 8504
```

---


