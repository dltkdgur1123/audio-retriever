# 환경 세팅
from dotenv import load_dotenv
import os
import time

# 스트리밋
import streamlit as st

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

# Chrominum의 headlesss instance를 사용하여 HTML 페이지 스크래핑
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage.file_system import LocalFileStore

from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.base import RunnableLambda

import openai
import asyncio
import subprocess
import math
from pydub import AudioSegment
import glob


# =====================================
# 필수 설정
# =====================================

# Window 환경에서만 설정(맥은 X)
# 비동기 소켓 처리 안정적처리 하기 위한 설정
if hasattr(asyncio, "WindowsProactorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    print(api_key[:20])
# 실행시간 출력
print(f"{os.path.basename(__file__)} 실행됨 {time.strftime('%Y-%m-%d %H:%M:%S')}")

# =====================================
# 필수 함수
# =====================================
llm = ChatOpenAI(temperature=0.1)

# 파일의 경로만 *.py
file_dir = os.path.dirname(os.path.realpath(__file__))

# ./cache ← 사용자들이 업로드한 파일 저장 (비디오 파일, 오디오 파일)
# ./chche/chunks ← 분할한 오디오 파일 저장 
upload_dir = os.path.join(file_dir, ".cache/chunks")
os.makedirs(upload_dir, exist_ok=True)

# 임베딩 경로
embedding_dir = os.path.join(file_dir, r"./.cache/embeddings")


# 오디오 추출 함수
@st.cache_resource()
def extract_audio_from_video(video_path, audio_path):
    command = ["ffmpeg", "-i", video_path, "-vn", audio_path, "-y"]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode !=0:
        st.error("오디오 추출에 실패했습니다.")
        st.code(result.stderr)
        return False

    return os.path.exists(audio_path)
  
  
# 원본 오디오를 chunk 쪼개는 함수 (10분 단위)
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = 1000 * 60 * chunk_size
    chunks = math.ceil(len(track) / chunk_len)
    
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        
        chunk = track[start_time:end_time]
        exp_path = os.path.join(chunks_folder, f"chunk_{i}.mp3")
        chunk.export(exp_path, format="mp3")


# 녹취록 -> 텍스트 파일로 저장하는 함수 
def transcribe_chunks(chunk_folder, destination):
    files = glob.glob(os.path.join(chunk_folder, "chunk*.mp3"))
    files.sort()
    
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a", encoding="utf-8") as text_file:
            print(file , "녹취록을 가져오는 중...", end="")
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                # language="en"
            )
            
            text_file.write(transcript.text)

# =====================================
# file_load  & cache
# =====================================


# splitter
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,    
)

# 임베딩
@st.cache_resource(show_spinner="Embedding file...") # 다시 호출하지 않기 위해
def embed_file(file_path, embedding_dir):
    cache_dir = LocalFileStore(embedding_dir)
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cache_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vector_store = FAISS.from_documents(docs, cache_embeddings)
    retriever = vector_store.as_retriever()
    return retriever

# =====================================
# Streamlit
# =====================================

# 페이지 타이틀
st.set_page_config(
    page_title="오디오 추출 분할 GPT"
)

st.markdown("""
    비디오 오디오 추출 분할 GPT
    
    사이드바를 이용해서 비디오를 업로드 하세요.
    그리고 비디오에 대해 질문해보세요.
""")

# 비디오 파일 업로더
with st.sidebar:
    
    # 사용자가 업로드 가능한 비디오 확장자 제한
    video = st.file_uploader(
        label="Video",
        type=["mp4","avi","mkv","mov"]
    )
    
    if video:
        with st.status("비디오 불러오는 중... / Loading video...") as status:
            # 영상별 독립 작업공간 생성
            
            # 영상 이름, 확장자 추출 ex) 파일명.mp4 -> 파일명 , .mp4
            video_name = os.path.splitext(video.name)[0]
            video_ext = os.path.splitext(video.name)[1]
            
            # 현재 영상 전용 cache 작업공간 생성 ex) .cache/파일명/
            video_cache_dir = os.path.join(file_dir, ".cache", video_name)
            
            # 작업공간 폴더 생성
            os.makedirs(video_cache_dir, exist_ok=True)
            
            # 파일 경로 생성 ->video, audio, transcript
            video_path = os.path.join(video_cache_dir, video.name)
            audio_path = os.path.join(video_cache_dir, "audio.mp3")
            transcript_path = os.path.join(video_cache_dir, "transcript.txt")
                        
            # 저장 폴더 생성
            chunk_folder = os.path.join(video_cache_dir, "chunk")
            embedding_dir = os.path.join(video_cache_dir, "embeddings")
            
            os.makedirs(chunk_folder, exist_ok=True) 
            os.makedirs(embedding_dir, exist_ok=True)
            
            # 읽어오기
            video_content = video.read()
            
                        
            # 저장하기
            with open(video_path, "wb") as f:
                f.write(video_content)
                
            status.update(label="영상에서 오디오 추출 중... / Extracting audio...")
            audio_created = extract_audio_from_video(video_path, audio_path)
            
            if not audio_created:
                st.error(f"오디오 파일이 생성되지 않았습니다: {audio_path}")
                st.stop()
        
            
            # 10분 단위로 스플릿
            status.update(label="영상에서 오디오 추출 중... / Extracting audio...")

            audio_created = extract_audio_from_video(video_path, audio_path)

            if not audio_created:
                st.error(f"오디오 파일이 생성되지 않았습니다: {audio_path}")
                st.stop()

            status.update(label="오디오를 구간별로 분할하는중... / Cutting audio Segments...")

            cut_audio_in_chunks(audio_path, 10, chunk_folder)
                        
            # 녹취록 추출            
            transcribe_chunks(chunk_folder, transcript_path)
            status.update(label="오디오를 텍스트로 변환중... / Transcripting Audio...") 
            
            # 작업 완료 상태 표시
            status.update(label="추출 완료! / completed!", state = "complete")
            
            
        # 3개 tab 메뉴
        transcript_tab, summary_tab, qa_tab = st.tabs(["Transcript","Summary", "QnA"])
        
        # 1. Transcript
        # 파일을 잘 가져왔는지 확인
        with transcript_tab:
            with open(transcript_path, "r", encoding="utf-8") as file:
                st .write(file.read())
        
        # 2. Summary (요약) 체인
        # refine document chain
        with summary_tab:
            start = st.button("요약하기")
            
            if start:
                loader = TextLoader(transcript_path, encoding="utf-8")
                docs = loader.load_and_split(text_splitter=splitter)
                
              
                # st.write(len(docs))
                # st.write(docs)
                
                first_summary_prompt = ChatPromptTemplate.from_template("""
                    다음내용을 텍스트를 짧고 핵심적으로 요약해줘 : {text}
                    길게 설명하지 말고 핵심만 짧게 요약해줘 : {text}                    
                """)
                
                # StrOutputParser(): 최종 결과를 문자열로 리턴
                first_summary_chain = first_summary_prompt | llm | StrOutputParser()
                
                summary = first_summary_chain.invoke({"text": docs[0].page_content})
                
                refine_prompt = ChatPromptTemplate.from_template("""
                    아래에 있는 기존 요약을 기준으로 최종 요약을 한국어로 만들어줘.
                    기존 요약:
                    {existing_summary}
                    추가로 참고할 내용:
                    ------------
                    {context}
                    ------------
                    추가 내용에 중요한 정보가 있다면 기존 요약에 자연스럽게 반영해서 수정해줘.
                    만약 기존 요약을 수정할 필요가 없다면, 기존 요약을 그대로 반환해줘.
                """)
                
                refine_chain = refine_prompt | llm | StrOutputParser()
                
                with st.status("요약하는 중... / summarizing..." ) as status:
                    for i, doc in enumerate(docs[1:]):
                        status.update(label=f"문서 처리 중.. {i+1}/{len(docs) - 1}")
                        
                        summary = refine_chain.invoke({
                            "existing_summary": summary, # 이전 요약본
                            "context": doc.page_content # 현재 document
                        })
                    
                        st.write(f"각각의 답변: {summary}")
                    
                st.write(f"최종 답변: {summary}")
        
        with qa_tab:
            retriever = embed_file(transcript_path, embedding_dir)
            question = st.text_input(label="영상에 대해서 질문하세요.")
            
            qa_prompt = ChatPromptTemplate.from_template("""
                너는 영상 내용을 바탕으로 질문에 답변하는 AI야.
                
                아래 참고 내용만 사용해서 질문에 한국어로 대답해줘.
                참고 내용에 없는 내용은 지어내지 말고
                "영상 내용만으로는 알 수 없습니다." 라고 답변하줘.
                
                참고 내용 : {context}
                
                질문 : {question}
                
                답변 :
            
            """)
            
            qa_chain = qa_prompt | llm | StrOutputParser()
            
            if question:
                docs = retriever.invoke(question)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                answer = qa_chain.invoke({
                    "context": context,
                    "question": question
                })
            
                st.write(answer)