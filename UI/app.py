import streamlit as st
import requests
import time
import os
from dotenv import load_dotenv

load_dotenv(override=True)

API_URL = os.getenv("API_URL", "http://localhost:8001")
API_KEY = os.getenv("API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

st.set_page_config(
    page_title="RAG Assistant",
    layout="centered"
)

st.title("RAG Assistant")
st.caption("HR-Finance-IT Policy Question Answering")

#Health Check
try:
    health = requests.get(f"{API_URL}/health", headers=HEADERS, timeout=3)
    if health.status_code !=200:
        st.error("Backend is not ready")
        st.stop()
except Exception:
    st.error("Cannot connect to FastAPI backend")
    st.stop()
    
    
query = st.text_area(
    "Ask a question",
    placeholder="eg. What is the leave Policy during probation",
    height=100
)

enable_web = st.checkbox(
    "联网搜索兜底",
    value=False,
    help="当知识库无法找到相关答案时，自动使用网络搜索作为补充",
)

ask = st.button("Ask")

if ask and query.strip():
    with st.spinner("Thinking...."):
        start = time.time()
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"query": query, "enable_web_search": enable_web},
                headers=HEADERS,
                timeout=120
            )
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

        latency = time.time() - start

    if response.status_code != 200:
        st.error(response.text)
        st.stop()

    data = response.json()
    st.session_state.answer = data.get("answer", "No answer returned")
    st.session_state.latency = latency

if "answer" in st.session_state:
    st.subheader("Answer")
    st.write(st.session_state.answer)
    st.caption(f"Latency: {st.session_state.latency:.2f}s")
    
    