import logging_config  # noqa: F401 – must be first; patches Tornado for HTTP access logging

import os
import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

st.title("Nord Pool Data Chatbot")
st.write(
    "Ask questions about day‑ahead, intraday auction and continuous trading data. "
    "This chatbot uses an LLM (via OpenAI) connected to your CSV files to answer queries "
    "and suggest trading insights."
)

@st.cache_data
def load_data():
    dfs = []
    data_dir = "data"
    filenames = [
        "dayahead_prices.csv",
        "ida1_prices.csv",
        "ida2_prices.csv",
        "ida3_prices.csv",
        "intraday_continuous_vwap_qh.csv",
    ]
    for fname in filenames:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["source_file"] = fname
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

df = load_data()

if not df.empty:
    with st.expander("Preview data"):
        st.dataframe(df.head())

api_key = st.text_input("Enter your OpenAI API key", type="password")
user_question = st.text_input(
    "Ask a question about the data",
    placeholder="e.g., Which area had the highest IDA1 price yesterday?",
)

if st.button("Ask") and api_key and user_question:
    llm = ChatOpenAI(openai_api_key=api_key, temperature=0)
    agent = create_pandas_dataframe_agent(
        llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS
    )
    with st.spinner("Thinking..."):
        response = agent.run(user_question)
    st.write(response)
