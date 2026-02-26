
import streamlit as st
from app.config import EURI_API_KEY
from app.chat_utils import get_chat_model, ask_chat_model
from app.pdf_utils import extract_text_from_pdf
from app.ui import pdf_uploader
from app.vectorstore_utils import create_faiss_index, retrive_similar_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
import pandas as pd

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="Military Intelligence AI", page_icon="🪖", layout="wide")

# ---------------- STYLE ---------------- #
st.markdown("""
<style>
.chat-message {padding:1rem;border-radius:0.5rem;margin-bottom:1rem;}
.chat-message.user {background:#2b313e;color:white;}
.chat-message.assistant {background:#f0f2f6;color:black;}
.stButton>button {background:#ff4b4b;color:white;font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# ---------------- STATE ---------------- #
if "messages" not in st.session_state: st.session_state.messages = []
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "chat_model" not in st.session_state: st.session_state.chat_model = None

# ---------------- HEADER ---------------- #
st.markdown("<h1 style='text-align:center;color:#ff4b4b'>🪖 Military Intelligence System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#666;'>Threat Analysis + Aspect-Based Sentiment + Entity Extraction</p>", unsafe_allow_html=True)

# ---------------- ENGINES ---------------- #
THREAT_WORDS = ["attack","missile","bomb","terror","nuclear","strike","war","invasion","weapon","troops","cyber","border"]
ASPECTS = {
    "Attack":["attack","strike","missile","bomb"],
    "Defense":["defense","shield","protect"],
    "Weapons":["weapon","nuclear","missile","rifle"],
    "Cyber":["cyber","hack","network"],
    "Border":["border","territory"]
}
ENTITIES = ["china","india","pakistan","usa","russia","missile","nuclear","army","navy","airforce","drone"]

def threat_score(text):
    text = text.lower()
    score = sum(text.count(w) for w in THREAT_WORDS)
    level = "LOW"
    if score > 12: level = "HIGH"
    elif score > 6: level = "MEDIUM"
    return score, level

def aspect_analysis(text):
    text = text.lower()
    data = {}
    for a, words in ASPECTS.items():
        data[a] = sum(text.count(w) for w in words)
    return data

def entity_extraction(text):
    found = []
    text = text.lower()
    for e in ENTITIES:
        if e in text:
            found.append(e.title())
    return list(set(found))

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.markdown("### 📁 Upload Military Documents")
    uploaded_files = pdf_uploader()

    if uploaded_files and st.button("🚀 Process Documents"):
        with st.spinner("Processing..."):
            texts = [extract_text_from_pdf(f) for f in uploaded_files]

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = []
            for t in texts: chunks.extend(splitter.split_text(t))

            st.session_state.vectorstore = create_faiss_index(chunks)
            st.session_state.chat_model = get_chat_model(EURI_API_KEY)

            st.success("✅ Documents Ready")

# ---------------- CHAT ---------------- #
st.markdown("## 💬 Military Intelligence Chat")

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="👤" if m["role"]=="user" else "🤖"):
        st.markdown(m["content"])
        st.caption(m["timestamp"])

if prompt := st.chat_input("Ask about operations, threats, weapons, borders..."):
    ts = time.strftime("%H:%M")
    st.session_state.messages.append({"role":"user","content":prompt,"timestamp":ts})

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        st.caption(ts)

    if st.session_state.vectorstore:
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔍 Analyzing..."):
                docs = retrive_similar_documents(st.session_state.vectorstore, prompt)
                context = "\n".join([d.page_content for d in docs])

                system_prompt = f"""
You are a Military Intelligence Analyst.
Summarize, assess risks, and answer.

Context:
{context}

Question:
{prompt}

Answer:
"""
                response = ask_chat_model(st.session_state.chat_model, system_prompt)

                score, level = threat_score(context)
                aspects = aspect_analysis(context)
                entities = entity_extraction(context)

            st.markdown("### 🧠 Intelligence Answer")
            st.markdown(response)

            st.markdown("### ⚠ Threat Meter")
            st.progress(min(score/20,1.0))
            st.success(f"Threat Level: {level} | Score: {score}")

            st.markdown("### 📊 Aspect Analysis")
            df = pd.DataFrame(aspects.items(), columns=["Aspect","Mentions"])
            st.bar_chart(df.set_index("Aspect"))

            st.markdown("### 🏷 Named Entities")
            st.write(entities)

            st.session_state.messages.append({"role":"assistant","content":response,"timestamp":ts})
    else:
        st.error("⚠ Please upload and process documents first!")

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.markdown("<center>🪖 Advanced Military Intelligence Platform</center>", unsafe_allow_html=True)
