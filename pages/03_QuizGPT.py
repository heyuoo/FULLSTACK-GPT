import json
import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser
from langchain.schema.runnable import RunnableMap


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

The difficulty level of the questions should be {difficulty}.

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
def run_quiz_chain(_docs, topic, difficulty):
    context = format_docs(_docs)

    chain = (
        RunnableMap(
            {
                "context": lambda _: context,
                "difficulty": lambda _: difficulty,
            }
        )
        | prompt
        | llm
    )
    response = chain.invoke({})
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

    difficulty = st.sidebar.selectbox(
        "Choose difficulty level", ("Easy", "Medium", "Hard")
    )
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
    show_correct_answers = st.checkbox("Show Correct Answers", value=True)

    api_key = os.getenv("OPENAI_API_KEY")

    if KeyError:
        api_key = st.sidebar.text_input(
            "Enter OpenAI API Key", type="password"
        )
    if not api_key:
        st.warning("API Key is required to proceed.")
        st.markdown(
            "[🚀View on"
            "Code](https://github.com/heyuoo/FULLSTACK-GPT/blob/streamlit5/app.py)"
        )
        st.stop()
    if len(api_key.strip()) <= 150:
        st.error("Invalid API Key. Please enter a valid OpenAI API Key.")
        st.markdown(
            "[🚀View on"
            "Code](https://github.com/heyuoo/FULLSTACK-GPT/blob/streamlit5/app.py)"
        )
        st.stop()
    else:
        st.sidebar.success("API Key loaded successfully!")
        st.markdown(
            "[🚀View on"
            "Code](https://github.com/heyuoo/FULLSTACK-GPT/blob/streamlit5/app.py)"
        )


if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name, difficulty)

    if response and "questions" in response:
        score = 0
        total_questions = len(response["questions"])

        with st.form("questions_form"):
            user_answers = []
            for idx, question in enumerate(response["questions"], start=1):
                st.write(f"Q {idx}. {question['question']}")
                value = st.radio(
                    "Select an option.",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                    key=f"question_{idx}",
                )
                user_answers.append((question, value))
                st.divider()
            submit_button = st.form_submit_button("Submit")

        if submit_button:
            for idx, (question, user_answer) in enumerate(
                user_answers, start=1
            ):

                if {"answer": user_answer, "correct": True} in question[
                    "answers"
                ]:
                    score += 1

            st.write(f"### Your score: {score}/{total_questions}")

            if score < total_questions:
                st.warning(
                    "You did not get a perfect score. Would you like to retry?"
                )
                retry_button = st.button("Retry")
                if retry_button:
                    for key in list(st.session_state.keys()):
                        if key.startswith("question_"):
                            del st.session_state[key]

            else:
                st.success("Perfect score! Well done!")
                st.balloons()

            for idx, (question, user_answer) in enumerate(
                user_answers, start=1
            ):
                correct_answers = [
                    answer["answer"]
                    for answer in question["answers"]
                    if answer["correct"]
                ]

                st.write(f"#### Q{idx}: {question['question']}")
                if {"answer": user_answer, "correct": True} in question[
                    "answers"
                ]:
                    st.success(f"Correct! Your answer: {user_answer}")
                elif user_answer is not None:
                    if show_correct_answers:
                        st.error(
                            f"Wrong! Your answer: {user_answer} | "
                            f"✔ Correct Answer: {correct_answers[0]}"
                        )
                    else:
                        st.error(f"Wrong! Your answer: {user_answer}")
