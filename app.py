from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import re
import csv
from datetime import datetime, timedelta
from typing import Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # 세션을 위한 시크릿 키

# 환경변수 로드
load_dotenv()

# 관리자 설정
ADMIN_PASSWORD = "1234"  # 실제 사용시 더 복잡한 비밀번호로 변경

# PDF 경로와 벡터 저장 디렉토리 설정
PDF_PATH = "noin3.pdf"
VECTOR_DIR = "vectorstore"
embeddings = OpenAIEmbeddings()

# 관리자 인증 데코레이터
def admin_required(f):
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# 챗봇 가드레일 클래스
class ChatbotGuardrails:
    def __init__(self):
        # 복지용구 관련 키워드
        self.welfare_keywords = [
            '복지', '용구', '신청', '등급', '부담', '자격', '품목', '보조', '지원',
            '노인', '장애', '의료', '재활', '보장', '수급', '급여', '서비스',
            '욕창', '매트리스', '방석', '보행기', '휠체어', '침대', '변기', '목욕',
            '산소', '호흡기', '발생기', '치료', '의료기기', '보장구'
        ]
        
        # 카테고리별 키워드 정의 (더 구체적으로)
        self.category_keywords = {
            '신청방법': ['신청방법', '신청 방법', '신청 절차', '신청 서류', '신청서', '제출', '접수', '처리', '어떻게 신청', '신청하려면'],
            '품목': ['품목', '종류', '제품', '기구', '장비', '보조기구', '재활용품', '어떤 것들', '품목에는', '종류에는'],
            '등급신청조건': ['등급', '등급 신청', '등급 조건', '등급 기준', '등급 판정', '등급 인정', '등급 요건', '자격조건', '신청 조건', '조건'],
            '본인부담률': ['본인부담률', '부담률', '본인 부담', '비용', '금액', '요금', '가격', '얼마', '비용', '할인', '부담'],
            '자격확인': ['자격', '자격 확인', '확인', '조사', '검토', '심사', '평가', '판단', '가능한지', '신청 가능']
        }
        
        # 금지 키워드 (복지용구와 무관한 주제)
        self.forbidden_keywords = [
            '로또', '사랑', '연애', '결혼', '주식', '투자', '게임', '음악', '영화',
            '요리', '여행', '쇼핑', '운동', '다이어트', '화장품', '패션'
        ]
        
        # 예시 질문들 (초기 가이드라인)
        self.example_questions = [
            "복지용구 신청 방법이 궁금해요",
            "복지용구 품목에는 어떤 것들이 있나요?",
            "복지용구 등급 신청 조건은 어떻게 되나요?",
            "복지용구 본인부담률은 얼마인가요?",
            "복지용구 자격 확인은 어떻게 하나요?",
            "복지용구 신청 서류는 무엇이 필요한가요?",
            "복지용구 수급자 자격은 어떻게 되나요?",
            "복지용구 급여 서비스는 어떻게 받을 수 있나요?"
        ]
        
        # 사용자별 마지막 질문 추적 (중복 방지용)
        self.user_last_questions = {}
        self.user_last_timestamps = {}
    
    def validate_question(self, question: str, user_id: str = "default") -> Dict[str, Any]:
        """질문 유효성 검증"""
        question = question.strip()
        
        # 1. 길이 검증
        if len(question) <= 3:
            return {
                'valid': False,
                'message': '질문을 좀 더 구체적으로 작성해주세요. (예: "복지용구 신청 방법이 궁금해요")',
                'examples': self.get_random_examples(3)
            }
        
        # 2. 의미없는 단어 검증
        meaningless_patterns = [r'^[아어음그저]+$', r'^[?!]+$', r'^[가-힣]{1,2}$']
        for pattern in meaningless_patterns:
            if re.match(pattern, question):
                return {
                    'valid': False,
                    'message': '구체적인 질문을 해주세요. 복지용구와 관련된 궁금한 점이 있으시면 언제든 물어보세요!',
                    'examples': self.get_random_examples(3)
                }
        
        # 3. 금지 키워드 검증
        for keyword in self.forbidden_keywords:
            if keyword in question:
                return {
                    'valid': False,
                    'message': '죄송합니다. 저는 노인복지용구 관련 질문에만 답변할 수 있어요. 복지용구와 관련된 궁금한 점이 있으시면 언제든 물어보세요!',
                    'examples': self.get_random_examples(3)
                }
        
        # 4. AI 기반 관련성 검증은 RAG에서 처리하도록 제거
        # 이제 RAG가 답변을 못하면 자동으로 fallback 처리됨
        
        return {'valid': True, 'message': '질문이 유효합니다.'}
    
    def classify_question(self, question: str, status: str = 'success') -> str:
        """질문을 카테고리별로 분류"""
        if status == 'fallback' or status == 'blocked':
            return '차단된질문'
        
        question_lower = question.lower()
        category_scores = {}
        
        # 각 카테고리별 점수 계산
        for category, keywords in self.category_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in question_lower:
                    score += 1
            category_scores[category] = score
        
        # 가장 높은 점수의 카테고리 반환
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            if category_scores[best_category] > 0:
                return best_category
        
        # 명확한 카테고리가 없으면 기타로 분류
        return '기타'
    
    def check_duplicate_question(self, question: str, user_id: str) -> Dict[str, Any]:
        """중복 질문 검증 (임시 비활성화)"""
        return {'valid': True, 'message': '중복 검증 통과'}
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """텍스트 유사도 계산"""
        # 간단한 유사도 계산 (실제로는 더 정교한 알고리즘 사용)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0
    
    def get_random_examples(self, count: int = 3) -> list:
        """랜덤 예시 질문 반환"""
        import random
        return random.sample(self.example_questions, min(count, len(self.example_questions)))
    
    def get_welcome_examples(self) -> list:
        """환영 예시 질문 반환"""
        return self.example_questions[:5]
    
    def get_fallback_response(self, error_type: str) -> str:
        """오류 발생 시 대체 응답"""
        fallback_responses = {
            'search_error': '죄송합니다. 현재 검색에 문제가 있어요. 잠시 후 다시 시도해주세요.',
            'api_error': '죄송합니다. 서비스에 일시적인 문제가 있어요. 잠시 후 다시 시도해주세요.',
            'general_error': '죄송합니다. 예상치 못한 오류가 발생했어요. 잠시 후 다시 시도해주세요.'
        }
        return fallback_responses.get(error_type, fallback_responses['general_error'])

# 가드레일 인스턴스 생성
guardrails = ChatbotGuardrails()



def save_chat_log(question, answer, is_fallback=False):
    """채팅 로그를 CSV 파일에 저장"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "fallback" if is_fallback else "success"
    category = guardrails.classify_question(question, status)
    
    # 현재 파일의 디렉토리에서 CSV 파일 찾기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'chat_log.csv')
    
    # CSV 파일이 없으면 헤더와 함께 생성
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['timestamp', 'question', 'answer', 'status', 'category'])
        writer.writerow([timestamp, question, answer, status, category])

def read_chat_logs(limit=None, category=None):
    """채팅 로그를 읽어오기"""
    # 현재 파일의 디렉토리에서 CSV 파일 찾기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'chat_log.csv')
    
    if not os.path.exists(csv_path):
        return []
    
    logs = []
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # HTML 템플릿에서 필요한 모든 필드 포함
                simple_log = {
                    'timestamp': row.get('timestamp', ''),
                    'question': row.get('question', ''),
                    'answer': row.get('answer', ''),
                    'status': row.get('status', 'success'),
                    'category': row.get('category', '기타')
                }
                
                # 카테고리 필터링
                if category and simple_log['category'] != category:
                    continue
                    
                logs.append(simple_log)
        
        # 최신 순으로 정렬
        logs.reverse()
        
        if limit:
            logs = logs[:limit]
        
        return logs
    except Exception as e:
        return []

# 벡터스토어 초기화
def init_vectorstore():
    if os.path.exists(VECTOR_DIR):
        vectorstore = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
        print("✅ 기존 벡터스토어 로드 완료")
    else:
        print("🛠️ 벡터스토어를 새로 생성합니다...")
        
        # PDF 파일 경로 확인
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_path = os.path.join(current_dir, PDF_PATH)
        print(f"📁 PDF 경로: {pdf_path}")
        print(f"📁 PDF 파일 존재: {os.path.exists(pdf_path)}")
        
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        print(f"📄 PDF 로드 완료: {len(docs)} 페이지")
        
        # 첫 번째 페이지 내용 확인
        if docs:
            print(f"📝 첫 번째 페이지 내용 (처음 200자): {docs[0].page_content[:200]}...")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
        split_documents = text_splitter.split_documents(docs)
        print(f"✂️ 텍스트 분할 완료: {len(split_documents)} 청크")
        
        # 첫 번째 청크 내용 확인
        if split_documents:
            print(f"📝 첫 번째 청크 내용 (처음 200자): {split_documents[0].page_content[:200]}...")
        
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
        vectorstore.save_local(VECTOR_DIR)
        print("✅ 벡터스토어 저장 완료")
    return vectorstore

# 체인 초기화
def init_chain():
    vectorstore = init_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6, "score_threshold": 0.5})
    
    prompt = PromptTemplate.from_template(
        """너는 노인복지용구 전문 상담 챗봇이야. 사용자의 질문에 대해서 복지용구 관련 자료(context)를 참고해서,
어르신들이 이해하기 쉽고 읽기 편하게 한국어로 설명해줘.

답변 작성 시 반드시 다음 마크다운 형식을 정확히 사용해주세요:

**1. 제목과 섹션:**
- 메인 제목: **제목**
- 섹션 제목: **섹션명:**
- 예시: **본인부담률:** 또는 **신청 자격:**

**2. 강조 표현:**
- 중요한 숫자나 키워드: **15%** 또는 **복지용구**
- 핵심 내용: **반드시 확인해야 할 사항**

**3. 목록과 체크리스트:**
- 일반 목록: • 항목
- 체크리스트: ✅ 항목 (줄바꿈 없이)
- 경고사항: ⚠️ 항목 (줄바꿈 없이)
- 연락처: 📞 항목 (줄바꿈 없이)
- 번호 목록: 1️⃣ 항목 (줄바꿈 없이)

**4. 구조화된 답변 예시:**
**본인부담률:**

✅ **일반 대상자:** **15%**

✅ **저소득층:** **5%**

⚠️ **주의사항:** 소득 기준에 따라 달라질 수 있습니다.

📞 **문의:** 국민건강보험공단에 확인해 주세요.

**주의:** 위 예시는 참고용이며, 실제 답변에서는 "제목"이라는 단어를 사용하지 마세요.

**5. 어르신 친화적 표현:**
- 존댓말 사용
- 복잡한 용어는 쉬운 말로 설명
- 한 번에 너무 많은 정보 주지 않기
- 핵심만 간단명료하게

**6. 안전장치:**
- 잘 모르는 내용은 추측하지 말고 "확실하지 않으니 공단에 문의해 주세요"라고 안내
- 복지용구 명칭이나 수급 조건은 명확하게 말해줘

답변 작성 시 반드시 다음 규칙을 지켜주세요:
1. 각 섹션마다 줄바꿈을 넣어주세요
2. 목록은 각 항목마다 줄바꿈을 넣어주세요
3. 마크다운 문법을 정확히 사용해주세요 (**굵은 글씨**, ✅, ⚠️ 등)
4. 읽기 쉽도록 적절한 공백을 넣어주세요

#Context: 
{context}

#Question:
{question}

#Answer:"""
    )
    
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

# 전역 변수로 체인 저장
chain = init_chain()

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        password = request.form.get('password')
        if password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return redirect(url_for('admin_logs'))
        else:
            return render_template('admin_login.html', error='비밀번호가 올바르지 않습니다.')
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('home'))

@app.route('/admin/logs')
@admin_required
def admin_logs():
    logs = read_chat_logs(limit=100)
    
    # 간단한 통계 계산 (status, category 없이)
    total_questions = len(logs)
    
    return render_template('admin_logs.html', 
                         logs=logs, 
                         total_questions=total_questions,
                         successful=total_questions,
                         blocked_errors=0,
                         success_rate=100.0,
                         categories={})

@app.route('/admin/api/logs')
@admin_required
def admin_api_logs():
    category = request.args.get('category')
    logs = read_chat_logs(limit=100, category=category)
    return jsonify({'logs': logs})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '').strip()
    user_id = data.get('user_id', 'web_user')
    
    if not question:
        return jsonify({'answer': '질문을 입력해주세요.', 'is_fallback': True, 'success': False})
    
    # 가드레일 검증
    validation = guardrails.validate_question(question, user_id)
    if not validation['valid']:
        response = {
            'answer': validation['message'],
            'is_fallback': True,
            'success': False
        }
        if 'examples' in validation:
            response['examples'] = validation['examples']
        if validation.get('is_duplicate', False):
            response['is_duplicate'] = True
        
        save_chat_log(question, validation['message'], is_fallback=True)
        return jsonify(response)
    

    
    try:
        # RAG 체인 실행
        answer = chain.invoke(question)
        save_chat_log(question, answer, is_fallback=False)
        return jsonify({'question': question, 'answer': answer, 'success': True})
    except Exception as e:
        print(f"Error: {e}")
        fallback_msg = guardrails.get_fallback_response('search_error')
        save_chat_log(question, fallback_msg, is_fallback=True)
        return jsonify({'answer': fallback_msg, 'is_fallback': True, 'success': False})

@app.route('/examples', methods=['GET'])
def get_examples():
    examples = guardrails.get_welcome_examples()
    return jsonify({'examples': examples})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 