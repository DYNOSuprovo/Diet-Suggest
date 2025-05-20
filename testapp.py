import streamlit as st
import os
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
# Import message types for history reconstruction
from langchain_core.messages import HumanMessage, AIMessage

import google.generativeai as genai
import logging # Add logging for better error inspection
import string # Import string module for punctuation removal
import re # Import regex for regional preference extraction
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Application started.")

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- API Key Checks and Configuration ---
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please set it in your .env file.")
    logging.error("GEMINI_API_KEY not found.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

try:
    genai.configure(api_key=GEMINI_API_KEY)
    llm_gemini = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.5)
    logging.info("Google Generative AI configured successfully.")
except Exception as e:
    st.error(f"Failed to configure Google Generative AI: {e}")
    logging.error(f"Google GenAI Configuration Error: {e}")
    st.stop()

if not GROQ_API_KEY:
    st.warning("GROQ_API_KEY not found. Groq suggestions will be unavailable.")
    logging.warning("GROQ_API_KEY not found.")

# --- Vector Database Setup (ChromaDB) ---
try:
    logging.info("Attempting to load SentenceTransformer model for embeddings.")
    try:
        SentenceTransformer("all-MiniLM-L6-v2")
        logging.info("SentenceTransformer model 'all-MiniLM-L6-v2' is available.")
    except Exception as model_e:
        st.error(f"Failed to load SentenceTransformer model: {model_e}. Please check your internet connection or environment.")
        logging.error(f"SentenceTransformer model loading error: {model_e}")
        st.stop()

    embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={'normalize_embeddings': True}
    )
    logging.info("HuggingFaceEmbeddings initialized successfully.")

    chroma_db_directory = "db"
    if not os.path.exists(chroma_db_directory):
        st.error(f"ChromaDB directory '{chroma_db_directory}' not found. Please ensure the DB is initialized first.")
        logging.error(f"ChromaDB directory '{chroma_db_directory}' not found.")
        st.stop()

    db = Chroma(persist_directory=chroma_db_directory, embedding_function=embedding)
    logging.info("Chroma DB loaded successfully.")

except Exception as e:
    st.error(f"VectorDB setup error: {e}")
    logging.exception("Full VectorDB setup traceback:")
    st.stop()

# --- RAG Prompt Template ---
diet_prompt = PromptTemplate.from_template("""
You are an AI assistant specialized in Indian diet and nutrition.
Based on the following conversation history and the user's query, provide a simple, practical, and culturally relevant **{dietary_type}** food suggestion suitable for Indian users aiming for **{goal}**.
If a specific region like **{region}** is mentioned or inferred, prioritize food suggestions from that region.
Focus on readily available ingredients and common Indian dietary patterns for the specified region.
Be helpful, encouraging, and specific where possible.
Use the chat history to understand the context of the user's current query and maintain continuity.
Strictly adhere to the **{dietary_type}** and **{goal}** requirements, and the **{region}** preference if specified.

Chat History:
{chat_history}

Context from Knowledge Base:
{context}

User Query:
{question}

{dietary_type} {goal} Food Suggestion (Tailored for {region} Indian context):
""")
logging.info("RAG Prompt template created.")

# --- Retrieval QA Chain Setup ---
try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_gemini,
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        chain_type="stuff",
        return_source_documents=True
    )
    logging.info("Retrieval QA Chain initialized successfully.")
except Exception as e:
    st.error(f"QA Chain setup error: {e}")
    logging.exception("Full QA Chain setup traceback:")
    st.stop()

# --- Session History Management ---
store = {}
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        logging.info(f"Creating new Langchain session history in 'store' for: {session_id}")
        store[session_id] = ChatMessageHistory()
    else:
        logging.info(f"Retrieving existing Langchain session history from 'store' for: {session_id}")
    return store[session_id]

# --- Conversational QA Chain Setup ---
conversational_qa_chain = RunnableWithMessageHistory(
    qa_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
    output_messages_key="answer"
)
logging.info("Conversational QA Chain initialized.")

# --- Merge Prompt Templates ---
merge_prompt_template_default = """
You are a diet planning assistant.
Your goal is to synthesize information from a primary RAG-based answer and several other AI suggestions into a single, coherent, and practical **{dietary_type}** diet plan or suggestion for **{goal}**, tailored for a **{region}** Indian context if a region is specified.
Prioritize the Primary RAG Answer. If it's weak or irrelevant, use Additional Suggestions.
Ensure the final plan is clear, actionable, and tailored for Indian users. Present as a clear list or paragraph.
If the user's input was *only* a greeting, respond politely. For inputs that include a greeting but also contain a query, focus on answering the query.

Primary RAG Answer:
{rag}

Additional Suggestions:
- LLaMA Suggestion: {llama}
- Mixtral Suggestion: {mixtral}
- Gemma Suggestion: {gemma}

Refined and Merged {dietary_type} {goal} Diet Plan/Suggestion (Tailored for {region} Indian context):
"""
merge_prompt_default = PromptTemplate.from_template(merge_prompt_template_default)

merge_prompt_template_table = """
You are a diet planning assistant.
Your goal is to synthesize information from a primary RAG-based answer and several other AI suggestions into a single, coherent, and practical **{dietary_type}** diet plan or suggestion for **{goal}**, tailored for a **{region}** Indian context if a region is specified.
Prioritize the Primary RAG Answer. If it's weak or irrelevant, use Additional Suggestions.
Ensure the final plan is clear, actionable, and tailored for Indian users.
**You MUST present the final diet plan as a clear markdown table. Include columns for Meal, Food Items, and Notes/Considerations.**
If the user's input was *only* a greeting, respond politely. For inputs that include a greeting but also contain a query, focus on answering the query.

Primary RAG Answer:
{rag}

Additional Suggestions:
- LLaMA Suggestion: {llama}
- Mixtral Suggestion: {mixtral}
- Gemma Suggestion: {gemma}

Refined and Merged {dietary_type} {goal} Diet Plan/Suggestion (Tailored for {region} Indian context, in markdown table format):
"""
merge_prompt_table = PromptTemplate.from_template(merge_prompt_template_table)
logging.info("Merge Prompt templates created.")

# --- Groq API Integration ---
def groq_diet_answer(model_name: str, query: str, user_diet_preference: str = "any", user_diet_goal: str = "diet", user_region: str = "Indian") -> str:
    if not GROQ_API_KEY:
        return f"Groq API key not available."
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        groq_model_map = {"llama": "llama3-70b-8192", "mixtral": "mixtral-8x7b-32976", "gemma": "gemma2-9b-it"}
        actual_model_name = groq_model_map.get(model_name.lower(), model_name)
        prompt_content = f"User query: '{query}'. Provide a concise, practical **{user_diet_preference}** diet suggestion or food item for **{user_diet_goal}**, tailored for a **{user_region}** Indian context. Focus on readily available ingredients. Be brief."
        payload = {"model": actual_model_name, "messages": [{"role": "user", "content": prompt_content}], "temperature": 0.5, "max_tokens": 250}
        logging.info(f"Calling Groq API: {actual_model_name} for query: '{query}' ({user_diet_preference} {user_diet_goal}, {user_region})")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data and data.get('choices') and data['choices'][0].get('message'):
            return data['choices'][0]['message']['content']
        return f"No suggestion from {actual_model_name} (empty/malformed response)."
    except requests.exceptions.Timeout: return f"Timeout error from {model_name}."
    except requests.exceptions.RequestException as e: return f"Request error from {model_name}: {e}"
    except Exception as e: return f"Error from {model_name}: {e}"

@st.cache_data(show_spinner=False, ttl=3600)
def cached_groq_answers(query: str, diet_preference: str, diet_goal: str, region: str) -> dict:
    logging.info(f"Fetching cached Groq answers for query: '{query}', pref: '{diet_preference}', goal: '{diet_goal}', region: '{region}'")
    models = ["llama", "mixtral", "gemma"]
    results = {}
    if not GROQ_API_KEY: return {k: "Groq API key not available." for k in models}
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        future_to_model = {executor.submit(groq_diet_answer, name, query, diet_preference, diet_goal, region): name for name in models}
        for future in future_to_model:
            model_name = future_to_model[future]
            try: results[model_name] = future.result()
            except Exception as e: results[model_name] = f"Failed: {e}"
    return results

# --- Helper Functions ---
GREETINGS = ["hi", "hello", "hey", "namaste", "yo", "vanakkam", "bonjour", "salaam", "good morning", "good afternoon", "good evening"]
TASK_KEYWORDS = ["diet", "plan", "weight", "gain", "loss", "table", "format", "chart", "show", "give", "vegan", "veg", "non-veg", "non veg", "vegetarian", "nonvegetarian", "south", "north", "east", "west", "india", "bengali", "punjabi", "maharashtrian", "gujarati", "tamil", "kannada", "telugu", "malayalam", "kanyakumari", "odisha", "oriya", "bhubaneswar", "cuttack", "angul"]
FORMATTING_KEYWORDS = ["table", "tabular", "chart", "format", "list", "bullet", "points", "itemize", "enumerate"]

def is_greeting(query: str) -> bool:
    if not query: return False
    cleaned_query = query.translate(str.maketrans('', '', string.punctuation)).strip().lower()
    contains_task_keyword = any(keyword in cleaned_query for keyword in TASK_KEYWORDS)
    is_short_greeting = cleaned_query in GREETINGS and len(cleaned_query.split()) <= 3
    return is_short_greeting and not contains_task_keyword

def is_formatting_request(query: str) -> bool:
    """Checks if the query is primarily a formatting request."""
    if not query: return False
    cleaned_query = query.translate(str.maketrans('', '', string.punctuation)).strip().lower()
    words = cleaned_query.split()

    if not any(keyword in cleaned_query for keyword in FORMATTING_KEYWORDS):
        return False # No formatting keywords found

    # If it's a short query and contains formatting keywords
    if len(words) <= 5:
        # Count non-formatting/non-filler words
        non_formatting_words = 0
        for word in words:
            if word not in FORMATTING_KEYWORDS and word not in ["in", "a", "as", "give", "me", "show", "it", "that", "please", "can", "you"]:
                non_formatting_words +=1
        # If very few (0 or 1) other substantive words, it's likely a formatting request
        if non_formatting_words <= 1:
            logging.info(f"Query '{query}' identified as formatting request (short, specific keywords).")
            return True
    # Check if it *only* contains formatting keywords and very common fillers
    # This is a stricter check
    only_formatting_and_fillers = True
    for word in words:
        if word not in FORMATTING_KEYWORDS and word not in ["in", "a", "as", "give", "me", "show", "it", "that", "please", "can", "you", "the", "for", "my", "me"]:
            only_formatting_and_fillers = False
            break
    if only_formatting_and_fillers:
        logging.info(f"Query '{query}' identified as formatting request (only formatting/fillers).")
        return True

    return False


def extract_diet_preference(query: str) -> str:
    query_lower = query.lower()
    if "non-veg" in query_lower or "non veg" in query_lower or "nonvegetarian" in query_lower: return "non-vegetarian"
    if "vegan" in query_lower: return "vegan"
    if "veg" in query_lower or "vegetarian" in query_lower: return "vegetarian"
    return "any"

def extract_diet_goal(query: str) -> str:
    query_lower = query.lower()
    if "weight gain" in query_lower or "gain weight" in query_lower or "gain" in query_lower : return "weight gain"
    if "weight loss" in query_lower or "loss weight" in query_lower or "loss" in query_lower: return "weight loss"
    return "diet"

def extract_regional_preference(query: str) -> str:
    query_lower = query.lower()
    match = re.search(r"(south|north|west|east) india(?:n)?|bengali|punjabi|maharashtrian|gujarati|tamil|kannada|telugu|malayalam|kanyakumari|odisha|oriya|bhubaneswar|cuttack|angul", query_lower)
    if match: return " ".join([word.capitalize() for word in match.group(0).replace('indian', 'Indian').split()])
    return "Indian"

def contains_table_request(query: str) -> bool:
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in ["table", "tabular", "chart", "in a table", "in table format", "as a table"])

# --- Streamlit UI ---
st.set_page_config(page_title="🍱 Indian Diet Advisor", layout="wide")
with st.sidebar:
    st.markdown("👤 **Created by Lord d'Artagnan**")
    st.markdown("---")
    use_llms_toggle = st.toggle("🔄 Include expanded suggestions", value=True, help="Fetch from LLaMA, Mixtral, Gemma via Groq.")

st.title("🥗 Personalized Indian Diet Recommendation")
st.markdown("Ask anything related to Indian diet and health.")

# --- Session State Initialization ---
if 'session_id' not in st.session_state:
    st.session_state.session_id = "session_" + os.urandom(8).hex()
    st.session_state.messages = []
    st.session_state.dietary_type = "any"
    st.session_state.diet_goal = "diet"
    st.session_state.region = "Indian"
    st.session_state.table_format_requested = False
    st.session_state.last_substantive_query = "" # Initialize new state variable
    get_session_history(st.session_state.session_id)
    logging.info(f"New Streamlit session: {st.session_state.session_id}")
else:
    get_session_history(st.session_state.session_id) # Ensure history is loaded
    if 'dietary_type' not in st.session_state: st.session_state.dietary_type = "any"
    if 'diet_goal' not in st.session_state: st.session_state.diet_goal = "diet"
    if 'region' not in st.session_state: st.session_state.region = "Indian"
    if 'table_format_requested' not in st.session_state: st.session_state.table_format_requested = False
    if 'last_substantive_query' not in st.session_state: st.session_state.last_substantive_query = ""
    logging.info(f"Existing session: {st.session_state.session_id}. Type: {st.session_state.dietary_type}, Goal: {st.session_state.diet_goal}, Region: {st.session_state.region}, Table: {st.session_state.table_format_requested}, LastQuery: {st.session_state.last_substantive_query}")

session_id_input = st.text_input("Session ID:", value=st.session_state.session_id, help="Change to switch conversation.")

if session_id_input and session_id_input != st.session_state.session_id:
    old_session_id = st.session_state.session_id
    st.session_state.session_id = session_id_input
    logging.info(f"User changed session ID from {old_session_id} to: {st.session_state.session_id}")
    current_langchain_history = get_session_history(st.session_state.session_id)
    new_ui_messages = []
    # Reset session specifics for the new/switched session
    st.session_state.dietary_type = "any"
    st.session_state.diet_goal = "diet"
    st.session_state.region = "Indian"
    st.session_state.table_format_requested = False
    st.session_state.last_substantive_query = "" # Reset for new session

    temp_last_substantive_query = ""
    for message_obj in current_langchain_history.messages:
        if isinstance(message_obj, HumanMessage):
            new_ui_messages.append({"role": "user", "content": message_obj.content})
            # Infer states from historical messages
            hist_diet_type = extract_diet_preference(message_obj.content)
            if hist_diet_type != "any": st.session_state.dietary_type = hist_diet_type
            hist_diet_goal = extract_diet_goal(message_obj.content)
            if hist_diet_goal != "diet": st.session_state.diet_goal = hist_diet_goal
            hist_region = extract_regional_preference(message_obj.content)
            if hist_region != "Indian": st.session_state.region = hist_region
            if contains_table_request(message_obj.content): st.session_state.table_format_requested = True
            
            # Update last substantive query from history
            if not is_formatting_request(message_obj.content) and not is_greeting(message_obj.content):
                temp_last_substantive_query = message_obj.content
        elif isinstance(message_obj, AIMessage):
            new_ui_messages.append({"role": "assistant", "content": message_obj.content})
    st.session_state.last_substantive_query = temp_last_substantive_query
    st.session_state.messages = new_ui_messages
    logging.info(f"Switched to session {st.session_state.session_id}. Loaded {len(new_ui_messages)} UI messages. Last substantive: '{temp_last_substantive_query}'")
    st.toast(f"Switched to session: {st.session_state.session_id}. History loaded.")
    st.rerun()

if "messages" not in st.session_state: st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Enter your diet-related question:", key="user_query_input")

if query:
    logging.info(f"Input: '{query}' for session: {st.session_state.session_id}")
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    pure_greeting = is_greeting(query)
    if pure_greeting:
        logging.info("Query is a pure greeting.")
        greeting_response = "Namaste! I am your diet planning assistant. How may I help you today?"
        st.session_state.messages.append({"role": "assistant", "content": greeting_response})
        with st.chat_message("assistant"): st.markdown(greeting_response)
        get_session_history(st.session_state.session_id).add_ai_message(greeting_response)
    else:
        logging.info("Query is NOT a pure greeting. Processing...")
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Determine the query to use for RAG and Groq
                query_for_rag_and_groq = query
                if is_formatting_request(query) and st.session_state.last_substantive_query:
                    query_for_rag_and_groq = st.session_state.last_substantive_query
                    logging.info(f"Identified formatting request. Using last substantive query for RAG/Groq: '{query_for_rag_and_groq}'")
                else:
                    # This is a new substantive query or no prior substantive query exists
                    st.session_state.last_substantive_query = query
                    logging.info(f"This is a substantive query. Updating last_substantive_query to: '{query}'")

                # Extract preferences from the *current* user query
                current_dietary_type = extract_diet_preference(query)
                current_diet_goal = extract_diet_goal(query)
                current_region = extract_regional_preference(query)
                current_table_request = contains_table_request(query)

                # Update session state based on *current* query's extractions
                if current_dietary_type != "any": st.session_state.dietary_type = current_dietary_type
                if current_diet_goal != "diet": st.session_state.diet_goal = current_diet_goal
                if current_region != "Indian": st.session_state.region = current_region
                if current_table_request: st.session_state.table_format_requested = True
                
                logging.info(f"Processing with: Query for RAG/Groq='{query_for_rag_and_groq}', SessionDietType='{st.session_state.dietary_type}', SessionGoal='{st.session_state.diet_goal}', SessionRegion='{st.session_state.region}', TableRequested='{st.session_state.table_format_requested}'")

                rag_answer = "Could not retrieve from knowledge base."
                try:
                    rag_result = conversational_qa_chain.invoke(
                        {"question": query_for_rag_and_groq, # Use potentially modified query
                         "dietary_type": st.session_state.dietary_type,
                         "goal": st.session_state.diet_goal,
                         "region": st.session_state.region},
                        config={"configurable": {"session_id": st.session_state.session_id}}
                    )
                    if isinstance(rag_result, dict) and "answer" in rag_result:
                        rag_answer = rag_result["answer"]
                        logging.info(f"RAG answer: '{rag_answer[:100]}...'")
                        if "source_documents" in rag_result: logging.info(f"Retrieved {len(rag_result['source_documents'])} sources.")
                    else:
                        logging.warning(f"RAG result unexpected format: {rag_result}")
                except Exception as e:
                    rag_answer = f"Error in RAG: {e}"
                    logging.exception("RAG chain error:")

                groq_suggestions = {"llama": "N/A", "mixtral": "N/A", "gemma": "N/A"}
                if use_llms_toggle and GROQ_API_KEY:
                    groq_suggestions = cached_groq_answers(query_for_rag_and_groq, st.session_state.dietary_type, st.session_state.diet_goal, st.session_state.region)
                elif not GROQ_API_KEY:
                    groq_suggestions = {k: "Groq API key not available." for k in groq_suggestions}


                final_answer = ""
                try:
                    current_merge_prompt = merge_prompt_table if st.session_state.table_format_requested else merge_prompt_default
                    logging.info(f"Using {'TABLE' if st.session_state.table_format_requested else 'DEFAULT'} merge prompt.")
                    
                    merge_input = {
                        "rag": rag_answer,
                        "llama": groq_suggestions.get("llama", "N/A"),
                        "mixtral": groq_suggestions.get("mixtral", "N/A"),
                        "gemma": groq_suggestions.get("gemma", "N/A"),
                        "dietary_type": st.session_state.dietary_type,
                        "goal": st.session_state.diet_goal,
                        "region": st.session_state.region
                    }
                    merge_chain = current_merge_prompt | llm_gemini
                    final_answer_obj = merge_chain.invoke(merge_input)
                    final_answer = final_answer_obj.content if hasattr(final_answer_obj, 'content') else str(final_answer_obj)
                    logging.info(f"Merged answer: '{final_answer[:100]}...'")
                except Exception as e:
                    final_answer = f"Error merging suggestions: {e}"
                    logging.exception("Merge process error:")

                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                
                # Update Langchain history with the final merged answer
                session_history = get_session_history(st.session_state.session_id)
                if session_history.messages and isinstance(session_history.messages[-1], AIMessage):
                    logging.info("Popping last RAG AI message from Langchain history to replace with final merged answer.")
                    session_history.messages.pop() 
                session_history.add_ai_message(final_answer)
                logging.info("Final merged answer added to Langchain history.")

st.markdown("---")
st.markdown("Disclaimer: This advisor provides general suggestions. Consult a healthcare professional for personalized advice.")
logging.info("Application request processing finished.")
