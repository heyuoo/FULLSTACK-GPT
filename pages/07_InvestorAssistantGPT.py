import streamlit as st
import openai as client
import json
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
import yfinance
import time


def get_ticker(inputs):
    ddg = DuckDuckGoSearchAPIWrapper()
    company_name = inputs["company_name"]
    return ddg.run(f"Ticker symbol of {company_name}")


def get_income_statement(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.income_stmt.to_json())


def get_balance_sheet(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.balance_sheet.to_json())


def get_daily_stock_performance(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.history(period="3mo").to_json())


functions_map = {
    "get_ticker": get_ticker,
    "get_income_statement": get_income_statement,
    "get_balance_sheet": get_balance_sheet,
    "get_daily_stock_performance": get_daily_stock_performance,
}


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


functions = [
    {
        "type": "function",
        "function": {
            "name": "get_ticker",
            "description": (
                "Given the name of a company returns its ticker symbol"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The name of the company",
                    }
                },
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_income_statement",
            "description": (
                "Given a ticker symbol (i.e AAPL) returns the company's income"
                " statement."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_balance_sheet",
            "description": (
                "Given a ticker symbol (i.e AAPL) returns the company's"
                " balance sheet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_stock_performance",
            "description": (
                "Given a ticker symbol (i.e AAPL) returns the performance of"
                " the stock for the last 100 days."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
]


st.set_page_config(
    page_title="InvestorAssistantGPT",
    page_icon="📈",
)


st.markdown(
    """
    ## Stock Analysis Assistant
            
    Welcome to InvestorAssistantGPT.
            
    Write down the name of a company and our Assistant will do the research for you.
"""
)


assistant_id = "asst_7Hy1vJizAZ25onqlFeuyfDTB"


if "messages" not in st.session_state:
    st.session_state["messages"] = []


company_name = st.chat_input(
    "Enter the name of the company you want to research :"
)

if company_name:

    if "thread" in st.session_state:
        del st.session_state["thread"]
    if "run" in st.session_state:
        del st.session_state["run"]

    paint_history()
    paint_message(company_name, "human")

    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": (
                    f"I want to know if the {company_name} stock is a good buy"
                ),
            }
        ]
    )
    st.session_state["thread"] = thread

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
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
