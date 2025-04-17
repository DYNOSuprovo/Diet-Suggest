import streamlit as st
import os
import requests
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from concurrent.futures import ThreadPoolExecutor

# Load .env keys
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set Gemini LLM
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
llm_gemini = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GEMINI_API_KEY)

# VectorDB + Embeddings
embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
db = Chroma(persist_directory="db", embedding_function=embedding)

# RAG Prompt
diet_prompt = PromptTemplate.from_template("""
Context:
{context}

User Query:
{question}

Generate a simple and practical food suggestion suited for Indian users.
""")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm_gemini,
    retriever=db.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": diet_prompt}
)

# Merge Prompt
merge_prompt = PromptTemplate.from_template("""
You are a diet planning assistant created by Lord d'Artagnan. Here is the RAG-based answer and additional suggestions.

🔹 RAG Answer:
{rag}

🔹 Other Suggestions:
{llama}
{mixtral}
{gemma}

Refine and merge these into ONE practical diet plan with clarity. Prioritize RAG.
Also handle greetings like hi, hello by introducing yourself and prompting the user politely.
""")

# Groq API call
def groq_diet_answer(model_name, query):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

# Cached parallel Groq model fetch
@st.cache_data
def cached_groq_answers(query):
    with ThreadPoolExecutor() as executor:
        futures = {
            "llama": executor.submit(groq_diet_answer, "llama-3.3-70b-versatile", query),
            "mixtral": executor.submit(groq_diet_answer, "mistral-saba-24b", query),
            "gemma": executor.submit(groq_diet_answer, "gemma2-9b-it", query),
        }
        return {k: f.result() for k, f in futures.items()}

# Streamlit UI
st.set_page_config(page_title="🍱 Diet Advisor", layout="centered")
st.title("🥗 Personalized Diet Recommendation")
st.markdown("Ask anything related to your diet and health.")

query = st.text_input("📥 Enter your diet-related question:", placeholder="E.g. I'm diabetic. What should I eat in lunch?")
use_llms = st.toggle("🔄 Include expanded suggestions", value=True)

# Greetings handling
greetings = ["hi", "hello", "hey", "namaste", "yo"]
if query.lower() in greetings:
    st.success("✅ Final Diet Plan")
    st.markdown("Bonjour, madame/monsieur! I am your diet planning assistant, created by the esteemed Lord d'Artagnan. 🤖\n\nHow may I assist you today?")
    st.stop()


# Process Query
if st.button("🔍 Get Diet Plan") and query:
    with st.spinner("🔍 Fetching core recommendation..."):
        rag = qa_chain.run(query)

    if use_llms:
        with st.spinner("⏳ Gathering additional suggestions from Groq..."):
            groq_ans = cached_groq_answers(query)
    else:
        groq_ans = {"llama": "", "mixtral": "", "gemma": ""}

    # Final merged response
    final_plan = llm_gemini.invoke(merge_prompt.format(
        rag=rag,
        llama=groq_ans["llama"],
        mixtral=groq_ans["mixtral"],
        gemma=groq_ans["gemma"]
    ))

    # Output Section
    st.subheader("📌 Core Suggestion")
    st.write(rag)

    if use_llms:
        st.subheader("📝 Additional Suggestions")
        st.write("**LLaMA 3:**", groq_ans["llama"])
        st.write("**Mixtral:**", groq_ans["mixtral"])
        st.write("**Gemma:**", groq_ans["gemma"])

    st.success("✅ Final Diet Plan")
    st.markdown(final_plan)
