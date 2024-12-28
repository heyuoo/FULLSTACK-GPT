import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain.schema import SystemMessage
from langchain.document_loaders import WebBaseLoader
from langchain.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from pydantic import BaseModel, Field
from typing import Any, Type
import os


# Define custom tools
class DuckDuckGoSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for")


class DuckDuckGoSearchTool(BaseTool):
    name = "DuckDuckGoSearchTool"
    description = (
        "Perform web searches using DuckDuckGo. Input a query string."
    )
    args_schema: Type[DuckDuckGoSearchToolArgsSchema] = (
        DuckDuckGoSearchToolArgsSchema
    )

    def _run(self, query) -> Any:
        search = DuckDuckGoSearchResults()
        return search.run(query)


wiki_api_wrapper = WikipediaAPIWrapper()


class WikipediaSearchToolArgsSchema(BaseModel):
    query: str = Field(
        description="The query you will search for on Wikipedia"
    )


class WikipediaSearchTool(BaseTool):
    name = "WikipediaSearchTool"
    description = "Perform searches on Wikipedia. Input a query string."
    args_schema: Type[WikipediaSearchToolArgsSchema] = (
        WikipediaSearchToolArgsSchema
    )

    def _run(self, query) -> Any:
        wiki = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)
        return wiki.run(query)


class WebScrapingToolArgsSchema(BaseModel):
    url: str = Field(description="The URL of the website you want to scrape")


class WebScrapingTool(BaseTool):
    name = "WebScrapingTool"
    description = "Scrape content from a given URL."
    args_schema: Type[WebScrapingToolArgsSchema] = WebScrapingToolArgsSchema

    def _run(self, url):
        loader = WebBaseLoader([url])
        docs = loader.load()
        return "\n\n".join([doc.page_content for doc in docs])


class SaveToTXTToolArgsSchema(BaseModel):
    text: str = Field(description="The text you will save to a file.")


class SaveToTXTTool(BaseTool):
    name = "SaveToTXTTOOL"
    description = """
    Use this tool to save the content as a .txt file.
    """
    args_schema: Type[SaveToTXTToolArgsSchema] = SaveToTXTToolArgsSchema

    def _run(self, text) -> Any:
        print(text)
        with open("research_results.txt", "w") as file:
            file.write(text)
        return "Research results saved to research_results.txt"


# Define Streamlit app
st.set_page_config(page_title="OpenAI Research Assistant", page_icon="📚")

st.title("OpenAI Research Assistant")
st.markdown(
    """
Use this assistant to gather information on any topic. 
It searches Wikipedia, DuckDuckGo, and other sources to provide detailed insights. Save findings to a .txt file.
"""
)

# Initialize the agent
llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")

system_message = SystemMessage(
    content="""
        You are a research expert. Use Wikipedia or DuckDuckGo to gather information.
        Scrape content from relevant links and provide comprehensive answers. Save findings to a .txt file.
    """
)

agent = initialize_agent(
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    tools=[
        DuckDuckGoSearchTool(),
        WikipediaSearchTool(),
        WebScrapingTool(),
    ],
    verbose=True,
    agent_kwargs={"system_message": system_message},
)

# Interactive input
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []

query = st.chat_input("Enter your research keyword :")

if query:

    result = agent.run(query)
    st.session_state["conversation"].append(
        {"query": query, "response": result}
    )


# Display conversation
if st.session_state["conversation"]:
    st.divider()

    for message in st.session_state["conversation"]:
        st.chat_message("user").markdown(message["query"])
        st.chat_message("assistant").markdown(message["response"])

# Option to save results
if st.session_state["conversation"]:
    if st.button("Save Results to File"):
        try:
            downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
            os.makedirs(downloads_dir, exist_ok=True)

            # Define file path
            def safe_filename(query):
                # 파일 이름에 사용할 수 없는 문자 제거 및 길이 제한
                invalid_chars = r'<>:"/\|?*'
                for char in invalid_chars:
                    query = query.replace(char, "")
                return (
                    query[:50] if len(query) > 50 else query
                )  # 최대 50자 제한

            # 파일 이름 설정
            filename = f"{(st.session_state['conversation'][-1]['query'])}.txt"
            save_path = os.path.join(downloads_dir, filename)

            with open(save_path, "w", encoding="utf-8") as file:

                file.write(
                    f"Query: {message['query']}\nResponse:"
                    f" {message['response']}\n\n"
                )
            st.success(f"Result saved successfully: {save_path}")
        except Exception as e:
            st.error(f"An error occurred while saving the file: {e}")
else:
    st.info("Perform research first. Enter your research keyword! ")
