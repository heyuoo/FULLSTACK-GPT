import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")


function = {
    "name": "get_questions",
    "description": (
        "function that takes a list of questions and answers and returns a"
        " quiz"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answer"],
                },
            },
        },
        "required": ["questions"],
    },
}


llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-0125",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
).bind(
    function_call={
        "name": "get_questions",
    },
    functions=[function],
)


prompt = PromptTemplate.from_template(
    """
You are a professional quiz creator who designs questions in Korean to test students' knowledge based on the given context.

You must create ten questions based on the information found in the provided context. Each question should have 4 options, with only one correct answer. All questions should be short and unique.



Context: {context}

"""
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"context": format_docs} | prompt | llm
    response = chain.invoke(_docs)
    arguments = json.loads(
        response.additional_kwargs["function_call"]["arguments"]
    )
    return arguments


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(topic):
    retriever = WikipediaRetriever(lang="ko")
    return retriever.get_relevant_documents(topic)


with st.sidebar:
    docs = None
    topic = None
    show_correct_answers = st.checkbox("Show Correct Answers", value=True)

    choice = st.selectbox(
        "Choose what you want to use.",
        ("File", "Wikipedia Article"),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name)

    if response and "questions" in response:

        with st.form("questions_form"):
            for idx, question in enumerate(response["questions"], start=1):
                st.write(f"Q {idx}. {question['question']}")
                value = st.radio(
                    "Select an option.",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                    key=f"question_{idx}",
                )
                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("Correct!")
                elif value is not None:
                    if show_correct_answers:
                        st.error("Wrong!")
                        correct_answers = [
                            answer["answer"]
                            for answer in question["answers"]
                            if answer["correct"]
                        ]
                        st.write("✔ Correct Answer : ", correct_answers[0])
                    else:
                        st.error("Wrong!")
                st.divider()
            button = st.form_submit_button()
