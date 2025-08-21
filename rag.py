import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# PDF 경로와 벡터 저장 디렉토리 설정
PDF_PATH = "noin2.pdf"
VECTOR_DIR = "vectorstore"
embeddings = OpenAIEmbeddings()

# ✅ 1. 벡터스토어 불러오기 또는 새로 생성
if os.path.exists(VECTOR_DIR):
    vectorstore = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
    print("✅ 기존 벡터스토어 로드 완료")
else:
    print("🛠️ 벡터스토어를 새로 생성합니다...")
    loader = PyMuPDFLoader(PDF_PATH)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    split_documents = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    vectorstore.save_local(VECTOR_DIR)
    print("✅ 벡터스토어 저장 완료")

# 2. 검색기 생성
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 3. 프롬프트 정의
prompt = PromptTemplate.from_template(
    """너는 노인복지용구 전문 상담 챗봇이야. 사용자의 질문에 대해서 복지용구 관련 자료(context)를 참고해서,
사용자의 질문에 대해 친절하고 이해하기 쉽게 마크다운(Markdown) 형식으로 정리해서 한국어로 대답해줘.

답변 시 다음 지침을 따르세요:
- **항목별로 번호 또는 불릿 리스트를 사용해서 보기 좋게 정리**
- **복지용구 명칭, 수급 조건은 명확하게 작성**
- **굵은 글씨, 인용구, 줄바꿈, 목록 등 마크다운 문법을 사용**
- **너무 긴 문장보다는 항목별로 짧고 명료하게**
- **확실하지 않은 내용은 "확실하지 않으니 국민건강보험공단에 문의해 주세요."라고 안내**
- 존댓말로, 어르신도 이해할 수 있는 말투로 설명해줘

#Context: 
{context}

#Question:
{question}

#Answer:"""
)

# 4. LLM 생성
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# 5. 체인 생성
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. 사용자 질문 반복
print("복지용구 챗봇입니다. 종료하려면 '종료' 또는 'exit'을 입력하세요.\n")

while True:
    user_input = input("질문: ")
    if user_input.strip().lower() in ["종료", "exit"]:
        print("챗봇을 종료합니다.")
        break
    try:
        answer = chain.invoke(user_input.strip())
        print("답변:", answer, "\n")
    except Exception as e:
        print("⚠️ 오류 발생:", e)
        print("죄송합니다. 다시 시도해 주세요.\n")
