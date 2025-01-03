from langchain.storage import LocalFileStore
import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import glob
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings


llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
)

has_transcript = os.path.exists("./.cache/podcast.txt")


splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
)


@st.cache_resource()
def embed_file(file_path):
    cache_dir = LocalFileStore(
        f"./.cache/embeddings/{os.path.basename(file_path)}"
    )
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if has_transcript:
        return
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(
            destination, "a", encoding="utf-8"
        ) as text_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="ko",
            )
            text_file.write(transcript.text)


@st.cache_data()
def extract_audio_from_video(video_path):
    if has_transcript:
        return
    audio_path = video_path.replace("mp4", "mp3")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        audio_path,
    ]
    subprocess.run(command)


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        chunk.export(
            f"{chunks_folder}/chunk_{str(i).zfill(2)}.mp3",
            format="mp3",
        )


st.set_page_config(
    page_title="MeetingGPT",
    page_icon="💼",
)

st.markdown(
    """
# MeetingGPT
            
Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.

Get started by uploading a video file in the sidebar.
"""
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


with st.sidebar:
    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mkv", "mov"],
    )

if video:
    chunks_folder = "./.cache/chunks"
    with st.status("Loading video...") as status:
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        transcript_path = video_path.replace("mp4", "txt")
        with open(video_path, "wb") as f:
            f.write(video_content)
        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path)
        status.update(label="Cutting audio segments...")
        cut_audio_in_chunks(audio_path, 10, chunks_folder)
        status.update(label="Transcribing audio...")
        transcribe_chunks(chunks_folder, transcript_path)

    transcript_tab, summary_tab, qa_tab = st.tabs(
        [
            "Transcript",
            "Summary",
            "Q&A",
        ]
    )

    with transcript_tab:
        with open(transcript_path, "r", encoding="utf-8") as file:
            st.write(file.read())

    with summary_tab:
        start = st.button("Generate summary")

        if start:

            loader = TextLoader(transcript_path, encoding="utf-8")

            docs = loader.load_and_split(text_splitter=splitter)
            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                다음 내용을 간결하게 요약하세요:
                     "{text}"
                간결한 요약:                
            """
            )
            first_summary_chain = (
                first_summary_prompt | llm | StrOutputParser()
            )
            summary = first_summary_chain.invoke(
                {"text": docs[0].page_content},
            )
            refine_prompt = ChatPromptTemplate.from_template(
                """
                당신의 작업은 최종 요약을 작성하는 것입니다.
    우리는 특정 지점까지 작성된 기존 요약을 제공했습니다: {existing_summary}
    아래의 추가 내용을 사용하여 기존 요약을 (필요한 경우에만) 보완하세요.
    ------------
    {context}
    ------------
    새로운 내용을 고려하여 기존 요약을 보완하세요.
    만약 추가 내용이 유용하지 않다면, 기존 요약을 반환하세요.
                """
            )
            refine_chain = refine_prompt | llm | StrOutputParser()
            with st.status("Summarizing...") as status:
                for i, doc in enumerate(docs[1:]):
                    status.update(
                        label=f"Processing document {i+1}/{len(docs)-1} "
                    )
                    summary = refine_chain.invoke(
                        {
                            "existing_summary": summary,
                            "context": doc.page_content,
                        }
                    )
                    st.write(summary)
            st.write(summary)

    with qa_tab:
        retriever = embed_file(transcript_path)
        question = st.text_input("Ask a question about the video:")
        docs = retriever.invoke(question)
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
            Your job is to answer the user's question using the given context.
            The context is a transcript from a video.
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
                ),
                ("human", "{question}"),
            ]
        )
        answer_chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | answer_prompt
            | llm
            | StrOutputParser()
        )
        answer = answer_chain.invoke(question)
        st.write(answer)
