from langchain.document_loaders import WebBaseLoader
from langchain.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
import json
import time
import streamlit as st
import openai as client


def duckduckgo_search(inputs):
    if "query" not in inputs:
        return "Error: Missing required parameter 'query'."
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
    if not url.startswith("http"):
        return "Error: Invalid URL."
    try:
        loader = WebBaseLoader([url])
        docs = loader.load()
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Error fetching content: {str(e)}"


def save_to_txt(inputs):
    content = inputs["content"]
    filename = inputs["filename"]
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)
    return f"Research results saved to {filename}"


functions_map = {
    "DuckDuckGoSearchTool": duckduckgo_search,
    "WikipediaSearchTool": wikipedia_search,
    "WebContentExtractorTool": web_scraping,
    "save_to_file": save_to_txt,
}


functions = [
    {
        "name": "DuckDuckGoSearchTool",
        "description": "Searches DuckDuckGo for a given query.",
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for DuckDuckGo.",
                }
            },
        },
    },
    {
        "name": "WikipediaSearchTool",
        "description": "Searches Wikipedia for a given query.",
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for Wikipedia.",
                }
            },
        },
    },
    {
        "name": "WebContentExtractorTool",
        "parameters": {
            "type": "object",
            "required": ["url"],
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "URL of the website to extract content from."
                    ),
                }
            },
        },
    },
    {
        "name": "save_to_file",
        "description": "Saves content to a .txt file.",
        "parameters": {
            "type": "object",
            "required": ["content", "filename"],
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Content to save to a file.",
                },
                "filename": {
                    "type": "string",
                    "description": "Filename to save the content to.",
                },
            },
        },
    },
]


def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content,
    )


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
        outputs.append(
            {
                "output": functions_map[function.name](
                    json.loads(function.arguments)
                ),
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


def validate_run(run_id, thread_id):
    try:
        run = get_run(run_id, thread_id)
        return run.status
    except client.NotFoundError:
        return None


# Define Streamlit app
st.set_page_config(page_title="OpenAI Research Assistant", page_icon="📚")

st.title("OpenAI Research Assistant")
st.markdown(
    """
Use this assistant to gather information on any topic. 
It searches Wikipedia, DuckDuckGo, and other sources to provide detailed insights. Save findings to a .txt file.
"""
)

assistant_id = "asst_Ff6rGIbsJiIsChAhacvYl5in"


if "messages" not in st.session_state:
    st.session_state["messages"] = []


query = st.chat_input("Enter your research keyword :")

if query:
    paint_history()
    paint_message(query, "human")

    if "thread" not in st.session_state:
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": f"I want to know about {query}.",
                }
            ]
        )
        st.session_state["thread"] = thread
    else:
        thread = st.session_state["thread"]

    if "run" in st.session_state:
        current_status = validate_run(st.session_state["run"].id, thread.id)
        if current_status in [None, "completed", "failed"]:
            del st.session_state["run"]
        else:
            st.error("Previous analysis is still in progress. Please wait.")
            st.stop()

    run = client.beta.threads.runs.create(
        thread_id=st.session_state["thread"].id,
        assistant_id=assistant_id,
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
                    get_tool_outputs(run.id, thread.id)
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


#    result = agent.run(query)
#    st.session_state["conversation"].append(
#        {"query": query, "response": result}
#    )
#
#
## Display conversation
# if st.session_state["conversation"]:
#    st.divider()
#
#    for message in st.session_state["conversation"]:
#        st.chat_message("user").markdown(message["query"])
#        st.chat_message("assistant").markdown(message["response"])
#
## Option to save results
# if st.session_state["conversation"]:
#    if st.button("Save Results to File"):
#        with open("research_results.txt", "w") as file:
#            for message in st.session_state["conversation"]:
#                file.write(
#                    f"Query: {message['query']}\nResponse:"
#                    f" {message['response']}\n\n"
#                )
#        st.success("Results saved to research_results.txt!")
# else:
#    st.info("Perform research first. Enter your research keyword! ")
#
