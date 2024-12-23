from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
import streamlit as st
import os


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


api_key = None


llm = ChatOpenAI(
    api_key=api_key,
    temperature=0.1,
)

memory = ConversationBufferMemory(
    llm=llm, memory_key="chat_history", return_messages=True
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]


answers_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    chat_history = inputs.get("chat_history", [])

    llm.streaming = False
    llm.callbacks = None
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question,
                        "context": doc.page_content,
                        "chat_history": chat_history,
                    }
                ).content,
                "source": doc.metadata["source"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    chat_history = inputs.get("chat_history", [])

    llm.streaming = True
    llm.callbacks = [ChatCallbackHandler()]

    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\n"
        for answer in answers
    )
    result = choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
            "chat_history": chat_history,
        }
    )
    return result


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


def find_history(query):
    histories = memory.load_memory_variables({})["chat_history"]
    temp = []
    for idx in range(len(histories) // 2):
        temp.append(
            {
                "input": histories[idx * 2].content,
                "output": histories[idx * 2 + 1].content,
            }
        )

    docs = [
        Document(
            page_content=f"input:{item['input']}\noutput:{item['output']}"
        )
        for item in temp
    ]
    try:
        vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
        found_docs = vector_store.similarity_search(query)
        candidate = found_docs[0].page_content.split("\n")[1]
        return candidate.replace("output:", "")
    except IndexError:
        return None


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/workers-ai\/).*",
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(
        docs, OpenAIEmbeddings(api_key=api_key)
    )
    return vector_store.as_retriever()


if "messages" not in st.session_state:
    st.session_state["messages"] = []


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )

    api_key = os.getenv("OPENAI_API_KEY")

    if KeyError:
        api_key = st.sidebar.text_input(
            "Enter OpenAI API Key", type="password"
        )
    if not api_key:
        st.warning("API Key is required to proceed.")
        st.markdown(
            "[🚀View on"
            "Code](https://github.com/heyuoo/FULLSTACK-GPT/blob/streamlit5/pages/03_SiteGPT.py)"
        )
        st.stop()
    if len(api_key.strip()) <= 150:
        st.error("Invalid API Key. Please enter a valid OpenAI API Key.")
        st.markdown(
            "[🚀View on"
            "Code](https://github.com/heyuoo/FULLSTACK-GPT/blob/streamlit5/pages/03_SiteGPT.py)"
        )
        st.stop()
    else:
        st.sidebar.success("API Key loaded successfully!")
        st.markdown(
            "[🚀View on"
            "Code](https://github.com/heyuoo/FULLSTACK-GPT/blob/streamlit5/pages/03_SiteGPT.py)"
        )


if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:

        retriever = load_website(url)
        send_message("I'm ready! Ask away!", "ai", save=False)
        paint_history()

        query = st.chat_input("Ask a question to the website.")

        if query:
            send_message(query, "human")

            found = find_history(query)
            if found:
                send_message(found, "ai")
            else:

                chain = (
                    {
                        "docs": retriever,
                        "question": RunnablePassthrough(),
                    }
                    | RunnablePassthrough.assign(chat_history=load_memory)
                    | RunnableLambda(get_answers)
                    | RunnableLambda(choose_answer)
                )
                with st.chat_message("ai"):
                    result = chain.invoke(query)
                    memory.save_context(
                        {"input": query},
                        {"output": result.content},
                    )


else:
    st.session_state["messages"] = []
