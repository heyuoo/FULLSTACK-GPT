from langchain.document_loaders import WebBaseLoader
from langchain.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
import json
import time
import os
import streamlit as st
import openai as client


def duckduckgo_search(inputs):

    query = inputs["query"]
    search = DuckDuckGoSearchResults()
    try:
        return search.run(query)
    except Exception as e:
        return f"Error occurred during DuckDuckGo search: {str(e)}"


def wikipedia_search(inputs):
    query = inputs["query"]
    wiki_api_wrapper = WikipediaAPIWrapper()
    wiki = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)
    return wiki.run(query)


def web_scraping(inputs):
    url = inputs.get("url", "")
    loader = WebBaseLoader([url])
    docs = loader.load()
    return "\n\n".join([doc.page_content for doc in docs])


def save_to_txt(inputs):
    content = inputs.get("text")
    download_folder = "./.cache/output"
    os.makedirs(download_folder, exist_ok=True)

    filename = "research_results.txt"
    file_path = os.path.join(download_folder, filename)

    counter = 1
    while os.path.exists(file_path):
        filename = f"research_results_{counter}.txt"
        file_path = os.path.join(download_folder, filename)
        counter += 1

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

    return {
        "message": f"Research results saved to {file_path}",
        "content": content,
        "filename": filename,
    }


functions_map = {
    "DuckDuckGoSearchTool": duckduckgo_search,
    "WikipediaSearchTool": wikipedia_search,
    "WebScrapingTool": web_scraping,
    "SaveToTXTTOOL": save_to_txt,
}


functions = [
    {
        "name": "DuckDuckGoSearchTool",
        "description": """
    Use this tool to perform web searches using the DuckDuckGo search engine.
    It takes a query as an argument.
    Example query: "Latest technology news"
    """,
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query you will search for string.",
                }
            },
        },
    },
    {
        "name": "WikipediaSearchTool",
        "description": """
    Use this tool to perform searches on Wikipedia.
    It takes a query as an argument.
    Example query: "Artificial Intelligence"
    """,
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The query you will search for on Wikipedia"
                    ),
                }
            },
        },
    },
    {
        "name": "WebScrapingTool",
        "description": """
    If you found the website link in DuckDuckGo,
    Use this to get the content of the link for my research.
    """,
        "parameters": {
            "type": "object",
            "required": ["url"],
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the website you want to scrape",
                }
            },
        },
    },
    {
        "name": "SaveToTXTTOOL",
        "description": """
    Use this tool to save the content as a .txt file.
    """,
        "parameters": {
            "type": "object",
            "required": ["content", "filename"],
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The text you will save to a file.",
                },
                "filename": {
                    "type": "string",
                    "description": "Filename to save the content to.",
                },
            },
        },
    },
]


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id, thread_id=thread_id
    )


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    return messages


def get_tool_outputs(run_id, thread_id):
    run = client.beta.threads.runs.retrieve(run_id=run_id, thread_id=thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        result = functions_map[function.name](json.loads(function.arguments))

        if isinstance(result, dict) or isinstance(result, list):
            result = json.dumps(result)
        outputs.append(
            {
                "output": result,
                "tool_call_id": action_id,
            }
        )

    return outputs


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs,
    )


def paint_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state["messages"]:
        paint_message(
            message["message"].replace("$", "\$"), message["role"], save=False
        )


st.set_page_config(page_title="OpenAI Research Assistant", page_icon="📚")

st.title("OpenAI Research Assistant")
st.markdown(
    """
Use this assistant to gather information on any topic. 
It searches Wikipedia, DuckDuckGo, and other sources to provide detailed insights. Save findings to a .txt file.
"""
)


assistants = client.beta.assistants.list(limit=10)
for a in assistants:
    if a.name == "Search Assistant":
        assistant = client.beta.assistants.retrieve(a.id)
        break
else:
    assistant = client.beta.assistants.create(
        name="Search Assistant",
        model="gpt-4o-mini",
        tools=functions,
        agent_kwargs={
            "system_message": """
        You are a research expert.

        Your task is to use Wikipedia or DuckDuckGo to gather comprehensive and accurate information about the query provided. 

        When you find a relevant website through DuckDuckGo, you must scrape the content from that website. Use this scraped content to thoroughly research and formulate a detailed answer to the question. 

        Combine information from Wikipedia, DuckDuckGo searches, and any relevant websites you find. Ensure that the final answer is well-organized and detailed, and include citations with links (URLs) for all sources used.

        Your research should be saved to a .txt file, and the content should match the detailed findings provided. Make sure to include all sources and relevant information.

        The information from Wikipedia must be included.

        Ensure that the final .txt file contains detailed information, all relevant sources, and citations.
        
        Please display the final content in the AI response.
        """
        },
    )

if "messages" not in st.session_state:
    st.session_state["messages"] = []


query = st.chat_input("Enter your research keyword :")

if query:
    if "thread" in st.session_state:
        del st.session_state["thread"]
    if "run" in st.session_state:
        del st.session_state["run"]

    paint_history()
    paint_message(query, "human")

    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": f"I want to know about {query}.",
            }
        ]
    )
    st.session_state["thread"] = thread

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    st.session_state["run"] = run

    with st.chat_message("ai"):
        with st.spinner("Analyzing..."):
            while get_run(run.id, thread.id).status in [
                "queued",
                "in_progress",
                "requires_action",
            ]:
                if get_run(run.id, thread.id).status == "requires_action":

                    submit_tool_outputs(run.id, thread.id)
                    time.sleep(0.5)
                else:
                    time.sleep(0.5)

            message = (
                get_messages(thread.id)[-1]
                .content[0]
                .text.value.replace("$", "\$")
            )

            st.session_state["messages"].append(
                {
                    "message": message,
                    "role": "ai",
                }
            )
            st.markdown(message)
