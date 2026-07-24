
import streamlit as st
import fitz

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os

from openai import OpenAI
from dotenv import load_dotenv

# -----------------------------
# 환경 설정
# -----------------------------
load_dotenv()

llm = ChatOpenAI(
    model="gpt-5.6-luna",
    temperature=0
)

# -----------------------------
# PDF 읽기 및 VectorDB 생성
# -----------------------------
@st.cache_resource
def create_vectorstore():

    pdf_doc = fitz.open("CNN NEWS_트럼프 8개국에 2050 관세 부과 경고.pdf")

    text = ""

    for page in pdf_doc:
        text += page.get_text()
        text += "\n\n"

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=30,
        chunk_overlap=10
    )

    chunks = text_splitter.split_text(text)

    vectorstore = FAISS.from_texts(
        chunks,
        OpenAIEmbeddings(model="text-embedding-3-large")
    )

    return vectorstore


vectorstore = create_vectorstore()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📚 CNN News Vocabulary AI")

st.write("CNN 뉴스를 기반으로 영어 단어를 설명해드립니다.")

query = st.text_input(
    "영어 단어를 입력하세요",
    placeholder="예) Tariffs"
)

# -----------------------------
# 검색 버튼
# -----------------------------
if st.button("검색"):

    if query == "":
        st.warning("영어 단어를 입력해주세요.")

    else:

        docs = vectorstore.similarity_search(query, k=5)

        context = ""

        for doc in docs:
            context += doc.page_content
            context += "\n\n"

        template = """
당신은 CNN 영어 뉴스 전문 영어 선생님입니다.

사용자가 영어 단어를 입력하면
다음 CNN 기사 내용을 참고하여 설명하세요.

배경지식

{context}

영어 단어

{question}

다음 형식으로 답하세요.

# 📚 영어 단어 학습

## 1. 단어 뜻

## 2. 품사

## 3. CNN 기사 예문

## 4. 문맥에서의 의미

## 5. 쉬운 설명

## 6. 새로운 예문

## 7. 비슷한 단어

## 8. 한국어 번역

답변은 Markdown 형식으로 작성하세요.
"""

        prompt = ChatPromptTemplate.from_template(template)

        chain = prompt | llm | StrOutputParser()

        with st.spinner("답변 생성 중..."):

            result = chain.invoke({
                "context": context,
                "question": query
            })

        st.markdown(result)

        with st.expander("📄 검색된 CNN 기사 내용"):

            st.write(context)
