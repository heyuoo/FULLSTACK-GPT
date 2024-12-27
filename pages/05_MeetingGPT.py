import streamlit as st
from langchain.schema import SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.document_loaders import WebBaseLoader
from typing import Type
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
import os


class WikipediaSearchToolArgsSchema(BaseModel):
    query: str = Field(description="Search query for Wikipedia.")


class WikipediaSearchTool(BaseTool):
    name: Type[str] = "WikipediaSearchTool"
    description: Type[str] = "Searches Wikipedia for a given query."
    args_schema: Type[WikipediaSearchToolArgsSchema] = (
        WikipediaSearchToolArgsSchema
    )

    def _run(self, query):
        wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        return wiki.run(query)


class DuckDuckGoSearchToolArgsSchema(BaseModel):
    query: str = Field(description="Search query for DuckDuckGo.")


class DuckDuckGoSearchTool(BaseTool):
    name: Type[str] = "DuckDuckGoSearchTool"
    description: Type[str] = "Searches DuckDuckGo for a given query."
    args_schema: Type[DuckDuckGoSearchToolArgsSchema] = (
        DuckDuckGoSearchToolArgsSchema
    )

    def _run(self, query):
        ddg = DuckDuckGoSearchRun()
        return ddg.run(query)


class WebContentExtractorArgsSchema(BaseModel):
    url: str = Field(description="URL of the website to extract content from.")


class WebContentExtractorTool(BaseTool):
    name: Type[str] = "WebContentExtractorTool"
    description: Type[str] = "Extracts text content from a given website URL."
    args_schema: Type[WebContentExtractorArgsSchema] = (
        WebContentExtractorArgsSchema
    )

    def _run(self, url):
        loader = WebBaseLoader(url)
        documents = loader.load()
        return "\n".join([doc.page_content for doc in documents])


class SaveToFileArgsSchema(BaseModel):
    content: str = Field(description="Content to save to a file.")
    filename: str = Field(description="Filename to save the content to.")


class SaveToFileTool(BaseTool):
    name: Type[str] = "SaveToFileTool"
    description: Type[str] = "Saves content to a .txt file."
    args_schema: Type[SaveToFileArgsSchema] = SaveToFileArgsSchema

    def _run(self, content, filename="output.txt"):
        target_dir = "output_files"
        os.makedirs(target_dir, exist_ok=True)
        save_path = os.path.join(target_dir, filename)
        with open(save_path, "w", encoding="utf-8") as file:
            file.write(content)
        return f"Content saved to {save_path}"


# Streamlit Setup
st.set_page_config(page_title="OpenAI Research Assistant", page_icon="🤖")

st.title("OpenAI Research Assistant")
st.markdown(
    """
This application uses OpenAI's assistant to gather information from various sources such as Wikipedia and DuckDuckGo, extract content from websites, and save it to files.
"""
)
# API Key Input
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

st.session_state.openai_api_key = st.text_input(
    "Enter your OpenAI API Key:", type="password", key="api_key"
)

if not st.session_state.openai_api_key:
    st.warning("Please enter your OpenAI API Key to proceed.")
    st.stop()

# Initialize Assistant
if "assistant" not in st.session_state:
    # Initialize OpenAI Chat Model
    llm = ChatOpenAI(
        openai_api_key=st.session_state.openai_api_key,
        temperature=0.1,
        model_name="gpt-4o-mini",
    )

    # Define the assistant agent
    st.session_state.assistant = initialize_agent(
        llm=llm,
        verbose=True,
        agent=AgentType.OPENAI_FUNCTIONS,
        tools=[
            WikipediaSearchTool(),
            DuckDuckGoSearchTool(),
            WebContentExtractorTool(),
            SaveToFileTool(),
        ],
        agent_kwargs={
            "system_message": SystemMessage(
                content="""You are an advanced OpenAI assistant. Your task is to gather information from various sources such as Wikipedia, DuckDuckGo, and websites. You can also save data into text files."""
            )
        },
    )
if "thread" not in st.session_state:
    st.session_state.thread = []

# User Input
query = st.text_input("Enter your research query:")

if "assistant_response" not in st.session_state:
    st.session_state.assistant_response = ""


if st.button("Start Research"):
    if query:
        st.session_state.thread.append(HumanMessage(content=query))
        with st.spinner("Researching..."):
            try:
                assistant_response = st.session_state.assistant.invoke(query)
                if isinstance(assistant_response, dict):
                    assistant_response_output = assistant_response.get(
                        "output", ""
                    )
                else:
                    assistant_response_output = str(assistant_response)

                st.session_state.thread.append(
                    SystemMessage(content=st.session_state.assistant_response)
                )
                st.session_state.assistant_response = assistant_response_output

                st.success("Query processed successfully!")

                st.text_area(
                    "Assistant Response",
                    st.session_state.assistant_response,
                    height=300,
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")

    else:
        st.warning("Please enter a query before starting research.")

        # Save result option

if st.session_state.assistant_response:
    if st.button("Save Result to File"):
        try:
            downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
            os.makedirs(downloads_dir, exist_ok=True)

            # Define file path
            filename = "research_result.txt"
            save_path = os.path.join(downloads_dir, filename)

            # Save assistant response to file
            with open(save_path, "w", encoding="utf-8") as file:
                file.write(st.session_state.assistant_response)

            # Display success message with file path
            st.success(f"Result saved successfully: {save_path}")

            # Optional: Provide a Streamlit download button

        except Exception as e:
            st.error(f"An error occurred while saving the file: {e}")
else:
    st.info("No response available to save. Perform research first.")
