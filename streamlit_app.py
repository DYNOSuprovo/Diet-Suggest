import streamlit as st
import os
import requests
from dotenv import load_dotenv
# Corrected imports from langchain_community
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from concurrent.futures import ThreadPoolExecutor
# sentence_transformers is used for the underlying model, keep this import
from sentence_transformers import SentenceTransformer
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import google.generativeai as genai
import logging # Add logging for better error inspection
import string # Import string module for punctuation removal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- API Key Checks and Configuration ---
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please set it in your .env file.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Use gemini-1.5-flash for potentially faster responses, adjust model if needed
    llm_gemini = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.7) # Added temperature
except Exception as e:
    st.error(f"Failed to configure Google Generative AI: {e}")
    st.stop()

if not GROQ_API_KEY:
    st.warning("GROQ_API_KEY not found. Groq suggestions will be unavailable.")


# --- Setup embeddings and vector DB ---
try:
    # Load the sentence transformer model
    # Using 'cpu' as specified, but note this can still sometimes interact poorly
    # with Streamlit's watcher. Ensure libraries are up to date.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # model.to("cpu") # This line might be redundant if specified in HuggingFaceEmbeddings
    logging.info("SentenceTransformer model loaded.")

    # Setup HuggingFaceEmbeddings
    embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}, # Explicitly set device
        encode_kwargs={'normalize_embeddings': True}
    )
    logging.info("HuggingFaceEmbeddings initialized.")

    # Check for the existence of the ChromaDB directory
    chroma_db_directory = "db"
    if not os.path.exists(chroma_db_directory):
        st.error(f"ChromaDB directory '{chroma_db_directory}' not found. Please ensure the DB is initialized.")
        st.stop()
    logging.info(f"ChromaDB directory '{chroma_db_directory}' found.")

    # Load the Chroma DB
    db = Chroma(persist_directory=chroma_db_directory, embedding_function=embedding)
    logging.info("Chroma DB loaded.")

except Exception as e:
    st.error(f"VectorDB setup error: {e}")
    # Consider logging the full traceback for debugging
    logging.exception("Full VectorDB setup traceback:")
    st.stop()

# --- Prompt for RAG ---
diet_prompt = PromptTemplate.from_template("""
You are an AI assistant specialized in Indian diet and nutrition.
Based on the following context and the user's query, provide a simple, practical, and culturally relevant food suggestion suitable for Indian users.
Focus on readily available ingredients and common Indian dietary patterns.
Be helpful, encouraging, and specific where possible.

Context:
{context}

User Query:
{question}

Food Suggestion (Tailored for Indian context):
""")
logging.info("RAG Prompt template created.")

# --- Setup QA Chain ---
try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_gemini,
        retriever=db.as_retriever(search_kwargs={"k": 3}), # Retrieve top 3 documents
        chain_type="stuff", # Use the 'stuff' chain type to stuff all documents into the prompt
        chain_type_kwargs={"prompt": diet_prompt},
        return_source_documents=True # Keep for potential debugging or display
    )
    logging.info("Retrieval QA Chain initialized.")
except Exception as e:
    st.error(f"QA Chain setup error: {e}")
    logging.exception("Full QA Chain setup traceback:")
    st.stop()

# --- Message History Setup ---
# In-memory store for session histories
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Gets the message history for a given session ID."""
    if session_id not in store:
        logging.info(f"Creating new session history for: {session_id}")
        store[session_id] = ChatMessageHistory()
    logging.info(f"Accessing session history for: {session_id}")
    return store[session_id]

# Wrap the QA chain with message history
conversational_qa_chain = RunnableWithMessageHistory(
    qa_chain,
    get_session_history,
    input_messages_key="query", # Key for the user's input in the chain
    history_messages_key="chat_history", # Key for chat history in the chain
    output_messages_key="answer" # Key for the chain's output
)
logging.info("Conversational QA Chain initialized with message history.")


# --- Merge Prompt ---
# This prompt guides the final synthesis by Gemini
merge_prompt_template = """
You are a diet planning assistant created by Lord d'Artagnan.
Your goal is to synthesize information from a primary RAG-based answer and several other AI suggestions into a single, coherent, and practical diet plan or suggestion.
Prioritize the Primary RAG Answer as the core information, but incorporate useful and relevant details from the Additional Suggestions if they enhance the practicality and Indian relevance of the advice.
Ensure the final plan is clear, actionable, and tailored for Indian users, using simple language and common food items.
Structure the response clearly. If the primary RAG answer is insufficient or seems off-topic, use the other suggestions to form a helpful response.

If the user's input was *only* a greeting (e.g., "hi", "hello!", "namaste."), respond politely by introducing yourself and asking how you can assist with their diet planning, instead of trying to generate a diet plan based on the information below. For inputs that include a greeting but also contain a query, focus on answering the query.

Primary RAG Answer:
{rag}

Additional Suggestions:
- LLaMA Suggestion: {llama}
- Mixtral Suggestion: {mixtral}
- Gemma Suggestion: {gemma}

Refined and Merged Diet Plan/Suggestion:
"""
merge_prompt = PromptTemplate.from_template(merge_prompt_template)
logging.info("Merge Prompt template created.")


# --- Groq API Integration ---
def groq_diet_answer(model_name: str, query: str) -> str:
    """Fetches a diet suggestion from a specified Groq model."""
    if not GROQ_API_KEY:
        logging.warning(f"Groq API key not available for {model_name}.")
        return f"Groq API key not available."
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        # Use more standard model names
        groq_model_map = {
            "llama": "llama3-70b-8192", # Corrected LLaMA model name
            "mixtral": "mixtral-8x7b-32768", # Corrected Mixtral model name
            "gemma": "gemma2-9b-it" # This one looks plausible
        }
        actual_model_name = groq_model_map.get(model_name.lower(), model_name)

        payload = {
            "model": actual_model_name,
            "messages": [{"role": "user", "content": f"User query: '{query}' Provide a concise, practical Indian vegetarian diet suggestion or food item for managing piles, considering regional relevance if possible. Be brief."}], # Simplified prompt for conciseness, added vegetarian and piles focus
            "temperature": 0.5, # Slightly lower temperature for more focused answers
            "max_tokens": 250 # Increased max tokens slightly
        }
        logging.info(f"Calling Groq API with model: {actual_model_name}")
        response = requests.post(url, headers=headers, json=payload, timeout=30) # Increased timeout
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        if data and data.get('choices') and len(data['choices']) > 0:
             return data['choices'][0]['message']['content']
        else:
             logging.warning(f"Groq API for {actual_model_name} returned empty response.")
             return f"No suggestion from {actual_model_name}."
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error from {model_name} ({actual_model_name}): {e}")
        return f"Request error from {model_name}: {e}"
    except Exception as e:
        logging.error(f"Unexpected error from {model_name} ({actual_model_name}): {e}")
        return f"Error from {model_name}: {e}"

# Cache results from Groq models
@st.cache_data(show_spinner=False, ttl=3600) # Cache for 1 hour
def cached_groq_answers(query: str) -> dict:
    """Fetches and caches suggestions from multiple Groq models concurrently."""
    logging.info(f"Fetching cached Groq answers for query: '{query}'")
    models = ["llama", "mixtral", "gemma"] # Use simple names for mapping
    results = {}
    if not GROQ_API_KEY:
        logging.warning("Groq API key not set, skipping Groq calls.")
        return {k: "Groq API key not available." for k in models}

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        # Submit tasks using the function that maps simple name to actual model name
        future_to_model = {executor.submit(groq_diet_answer, name, query): name for name in models}
        for future in future_to_model:
            model_name = future_to_model[future]
            try:
                results[model_name] = future.result()
                logging.info(f"Successfully fetched {model_name} suggestion.")
            except Exception as e:
                logging.error(f"Error fetching {model_name} suggestion: {e}")
                results[model_name] = f"Failed to get suggestion: {e}"
    return results

# Define greeting variations for the robust check
GREETINGS = ["hi", "hello", "hey", "namaste", "yo", "vanakkam", "bonjour", "salaam"]

def is_greeting(query: str) -> bool:
    """Checks if the query is purely a greeting (allows punctuation)."""
    if not query:
        return False
    # Remove punctuation and convert to lowercase for checking
    cleaned_query = query.translate(str.maketrans('', '', string.punctuation)).strip().lower()
    logging.info(f"Greeting check: Original query='{query}', Cleaned query='{cleaned_query}'")
    # Check if the cleaned query is exactly one of the predefined greetings
    is_pure_greeting = cleaned_query in GREETINGS
    logging.info(f"Greeting check: Is pure greeting? {is_pure_greeting}")
    return is_pure_greeting


# --- Streamlit UI ---
st.set_page_config(page_title="🍱 Indian Diet Advisor", layout="wide")
st.title("🥗 Personalized Indian Diet Recommendation")
st.markdown("Ask anything related to Indian diet and health. Suggestions are tailored for Indian users.")

# Session ID management
if 'session_id' not in st.session_state:
    st.session_state.session_id = "session_" + os.urandom(8).hex()

session_id_input = st.text_input("Session ID:", value=st.session_state.session_id, help="Use a unique ID for each conversation session.")
if session_id_input and session_id_input != st.session_state.session_id:
    st.session_state.session_id = session_id_input
    st.toast(f"Switched to session: {st.session_state.session_id}")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input area at the bottom
# Use a container if you want to add a border or background to the input area
# with st.container():
query = st.chat_input("Enter your diet-related question:", key="user_query_input") # Use st.chat_input for typical chat layout

# Options placed elsewhere (e.g., sidebar or settings expander)
# For simplicity, let's add the toggle here for now, but visually it might be better elsewhere.
use_llms_toggle = st.toggle("🔄 Include expanded suggestions from other models", value=True, help="Fetch additional suggestions from LLaMA, Mixtral, and Gemma via Groq API.")


# Trigger logic when user sends a message via st.chat_input
if query:
    logging.info(f"Input received via chat_input: '{query}'")

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message immediately (Streamlit reruns anyway, but this shows intent)
    with st.chat_message("user"):
        st.markdown(query)

    # Check if it's a pure greeting
    if is_greeting(query):
        logging.info("Query identified as a pure greeting.")
        greeting_response = "Bonjour! I am your diet assistant created by Lord d'Artagnan. How may I help you today?"
        st.session_state.messages.append({"role": "assistant", "content": greeting_response})
        with st.chat_message("assistant"):
            st.markdown(greeting_response)
        # st.rerun() # Optional: force rerun to clear the input box and display the new message
    else:
        logging.info("Query is NOT a pure greeting. Proceeding with RAG/Merge.")
        # Display a spinner while processing
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."): # Spinner appears in the assistant's message bubble
                # --- Get RAG Answer ---
                rag_answer = ""
                rag_sources = []

                try:
                    # Invoke the conversational chain
                    # The 'query' key must match input_messages_key in RunnableWithMessageHistory
                    result = conversational_qa_chain.invoke(
                        {"query": query},
                        config={"configurable": {"session_id": st.session_state.session_id}}
                    )
                    logging.info(f"RAG Chain raw result type: {type(result)}")
                    logging.info(f"RAG Chain raw result: {result}")

                    # Safely get the answer and sources
                    if isinstance(result, dict):
                        rag_answer = result.get("answer", "Could not retrieve a specific answer from the knowledge base.")
                        rag_sources = result.get("source_documents", [])
                        logging.info(f"Successfully extracted answer and sources from RAG result.")
                    else:
                        rag_answer = str(result) if result else "Could not retrieve a specific answer from the knowledge base."
                        rag_sources = []
                        logging.warning(f"RAG chain result was not a dict, got type: {type(result)}")

                except Exception as e:
                    rag_answer = f"Error retrieving answer from knowledge base: {e}"
                    logging.exception("Error during RAG chain invocation:")
                    # st.error(rag_answer) # Don't use st.error directly in the chat bubble, log instead

                # --- Get Additional Suggestions (if toggle is on and Groq key exists) ---
                llama = mixtral = gemma = ""
                if use_llms_toggle and GROQ_API_KEY: # Use the actual toggle state
                    # Spinner already active for "Thinking..." covering this.
                    groq_results = cached_groq_answers(query)
                    llama = groq_results.get("llama", "N/A")
                    mixtral = groq_results.get("mixtral", "N/A")
                    gemma = groq_results.get("gemma", "N/A")
                    logging.info(f"Fetched Groq results: Llama='{llama[:50]}...', Mixtral='{mixtral[:50]}...', Gemma='{gemma[:50]}...'")
                elif use_llms_toggle and not GROQ_API_KEY:
                    llama = mixtral = gemma = "Groq API key not available."
                    logging.warning("Groq toggle is on but key is missing.")
                else:
                    llama = mixtral = gemma = "Suggestions from other models are disabled."
                    logging.info("Groq toggle is off.")


                # --- Merge Suggestions using Gemini ---
                final_merged = ""
                try:
                    # Format the merge prompt with all collected information
                    final_prompt_text = merge_prompt.format(
                        rag=rag_answer,
                        llama=llama,
                        mixtral=mixtral,
                        gemma=gemma
                    )
                    logging.info("Merge prompt formatted.")
                    # Invoke Gemini with the formatted prompt
                    final_merged = llm_gemini.invoke(final_prompt_text)
                    logging.info("Gemini merge model invoked successfully.")

                except Exception as e:
                    final_merged = f"Error merging suggestions: {e}"
                    logging.exception("Error during final merge invocation:")
                    # st.error(final_merged) # Log instead


            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": final_merged})
            # Display assistant message immediately
            st.markdown(final_merged)

# Add some footer or explanation
st.markdown("---")
st.markdown("Disclaimer: This advisor provides general suggestions based on AI models and a knowledge base. Consult a qualified healthcare professional or dietitian for personalized medical or dietary advice.")

