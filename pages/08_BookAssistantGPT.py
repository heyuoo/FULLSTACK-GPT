import streamlit as st
import openai as client
import os
import time
import re

st.set_page_config(
    page_title="BookAssistantGPT",
    page_icon="📖",
)

st.title("File-Based Assistant")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload a file and ask questions based on its content.
"""
)


assistant_id = "asst_PMX0zlhX3Hfg2P35jsAvJK3g"


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=content
    )


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    return messages


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def paint_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        paint_message(
            message["message"],
            message["role"],
            save=False,
        )


with st.sidebar:
    uploaded_file = st.file_uploader(
        "Upload a .txt .pdf or .docx file", type=["pdf", "txt", "docx"]
    )

if uploaded_file:

    vector_store = client.beta.vector_stores.create(
        name=f"{uploaded_file.name}"
    )
    file_paths = [f"./files/{uploaded_file.name}"]
    file_streams = [open(path, "rb") for path in file_paths]

    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id,
        files=file_streams,
    )
    assistant = client.beta.assistants.update(
        assistant_id=assistant_id,
        tool_resources={
            "file_search": {"vector_store_ids": [vector_store.id]}
        },
    )
    message_file = client.files.create(
        file=open(f"./files/{uploaded_file.name}", "rb"), purpose="assistants"
    )

if "messages" not in st.session_state:
    st.session_state["messages"] = []


query = st.chat_input("Ask anything about this file")
if query:
    paint_history()
    paint_message(query, "human")
    if not st.session_state.get("thread"):
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "f'Please refer to the uploaded file for my"
                        " question:' query"
                    ),
                    "attachments": [
                        {
                            "file_id": message_file.id,
                            "tools": [{"type": "file_search"}],
                        }
                    ],
                }
            ]
        )
        send_message(thread.id, query)
        st.session_state["thread"] = [thread]
    else:
        thread = st.session_state["thread"][0]
        send_message(thread.id, query)
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )
    with st.chat_message("ai"):
        with st.spinner("Creating an answer..."):
            while get_run(run.id, thread.id).status in [
                "queued",
                "in_progress",
                "requires_action",
            ]:
                time.sleep(0.5)
        message = get_messages(thread.id)[0].content[0].text.value
        # Remove annotations like
        message = re.sub(r"【.*?】", "", message)
        # Save the cleaned message
        save_message(message, "ai")
        st.markdown(message)
else:
    st.session_state["messages"] = []
    st.session_state["thread"] = []
