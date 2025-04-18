import streamlit as st
import os
import requests
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer  # Important patch
import asyncio

# Load .env keys
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set Gemini LLM
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
llm_gemini = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GEMINI_API_KEY)

# VectorDB + Embeddings with patch to fix meta tensor error
model = SentenceTransformer("all-MiniLM-L6-v2")
model.to("cpu")

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
embedding.client = model

db = Chroma(persist_directory="db", embedding_function=embedding)

# RAG Prompt
diet_prompt = PromptTemplate.from_template("""
You are a nutrition assistant helping Indian users make healthy food choices.

Context:
{context}

User Query:
{question}

Instructions:
- Suggest **practical**, **affordable**, and **region-friendly** Indian foods.
- If the user mentions a condition (e.g., diabetes, BP), tailor accordingly.
- Use simple words and avoid medical jargon.
- Keep suggestions short and bullet-pointed.

Output Format:
- Meal: [Breakfast/Lunch/Dinner]
- Foods: [List items]
- Notes: [Any condition-based warning or tip]
""")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm_gemini,
    retriever=db.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": diet_prompt}
)

# Merge Prompt
merge_prompt = PromptTemplate.from_template("""
You are a personalized Indian diet planner created by Lord d'Artagnan. Your goal is to combine AI-generated responses into one clear, practical diet plan for Indian users.

User Question: (Implied from RAG and LLM responses)

🔹 RAG-Based Core Recommendation:
{rag}

🔹 Additional Suggestions from other expert AIs:
- LLaMA 3:
{llama}

- Mixtral:
{mixtral}

- Gemma:
{gemma}

Instructions:
1. Read and analyze all the suggestions above.
2. Prioritize the **RAG-based suggestion** first — it's backed by trusted data.
3. Use useful ideas from other models if they add variety or clarity.
4. Remove any conflicting advice — prefer simplicity over complexity.
5. Ensure cultural and health relevance for Indian users.
6. Handle greetings like "hi", "hello", "namaste" gracefully by introducing yourself and asking how you can help.

Output Format:
👋 Greeting (if applicable)

📌 Final Diet Plan:
- Breakfast: ...
- Lunch: ...
- Dinner: ...
- Tips: (Optional health or diet advice)

Keep it concise, friendly, and clear.
""")

# Groq API call (asynchronous)
async def groq_diet_answer(model_name, query):
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

# Fetch multiple answers asynchronously
async def fetch_groq_answers(query):
    models = ["llama-3.3-70b-versatile", "mistral-saba-24b", "gemma2-9b-it"]
    tasks = [groq_diet_answer(model, query) for model in models]
    responses = await asyncio.gather(*tasks)
    return {
        "llama": responses[0],
        "mixtral": responses[1],
        "gemma": responses[2]
    }

# Streamlit UI
st.set_page_config(page_title="🍱 Diet Advisor", layout="centered")
st.title("🥗 Personalized Diet Recommendation")
st.markdown("Ask anything related to your diet and health.")

query = st.text_input("📥 Enter your diet-related question:", placeholder="E.g. I'm diabetic. What should I eat in lunch?")
use_llms = st.checkbox("🔄 Include expanded suggestions", value=True)

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
            # Fetch additional suggestions asynchronously
            groq_ans = asyncio.run(fetch_groq_answers(query))
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
