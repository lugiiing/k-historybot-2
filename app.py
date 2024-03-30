import os
import openai
import streamlit as st

from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader


# 타이틀 적용, # 특수 이모티콘 삽입 예시
# emoji: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
st.title(':robot_face:한국사봇? 너 정말 똑똑하니?')

# 캡션 적용
st.caption('한국사 교과서를 읽고 이해하여 답변을 제공하는 로봇입니다. 로봇이 교과서를 제대로 이해했는지 확인해보세요!')

# 마크다운 부가설명
st.markdown('###### 질문, 요약 등 다양한 부탁을 해 보세요! 교과서의 어떤 부분을 참고했는지 비교하며 한국사봇:robot_face:이 교과서를 제대로 이해했는지 확인해보세요!:sparkles:')



api_key = st.text_input(label='OpenAI API 키를 입력하세요', type='password')



if api_key:
    # OpenAI API를 사용하기 위한 처리 과정을 함수로 정의
    def initialize_openai_processing(api_key):
        #client = OpenAI()
        #OpenAI.api_key = api_key


        loader = DirectoryLoader('./khistory_data', glob="*.txt", loader_cls=TextLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        
        persist_directory = 'db'
        #embedding = OpenAIEmbeddings()
        embedding = OpenAIEmbeddings(api_key=api_key)  # API 키를 생성자에 전달
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embedding,
            persist_directory=persist_directory)
        
        vectordb.persist()
        vectordb = None

        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding)
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        qa_chain = RetrievalQA.from_chain_type(
            #llm=OpenAI(),
            llm=OpenAI(api_key=api_key),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True)

        return embedding, vectordb, qa_chain

    # 함수 호출로 초기화 과정 수행
    embedding, vectordb, qa_chain = initialize_openai_processing(api_key)


    # 텍스트 입력
    query = st.text_input(
        label='한국사봇에게 질문해보세요!', 
        placeholder='예시: 동학 농민 운동은 왜 일어났나요?'
    )  
    
    # 버튼 클릭
    button = st.button(':robot_face:한국사봇에게 물어보기')

    if button:
        llm_response = qa_chain(query)
        #process_llm_response(llm_response)
        result = llm_response.get('result')
        source_documents1 = llm_response.get('source_documents')[0]
        source_documents2 = llm_response.get('source_documents')[1]
        source_documents3 = llm_response.get('source_documents')[2]
        st.write('결과: ', f'{result}')
        st.write('교과서 내용 1: 'f'{source_documents1}')
        st.write('교과서 내용 2: 'f'{source_documents2}')
        st.write('교과서 내용 3: 'f'{source_documents3}')
