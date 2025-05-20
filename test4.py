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
# Import message types for history reconstruction
from langchain_core.messages import HumanMessage, AIMessage

import google.generativeai as genai
import logging # Add logging for better error inspection
import string # Import string module for punctuation removal
import re # Import regex for regional preference extraction

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Application started.")

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- API Key Checks and Configuration ---
# Check if the Gemini API key is available, otherwise stop the app.
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please set it in your .env file.")
    logging.error("GEMINI_API_KEY not found.")
    st.stop() # Halts the Streamlit app execution.

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Use a slightly higher temperature for potentially more creative merging, but keep it reasonable
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
    # Ensure the model is downloaded if not already present
    try:
        SentenceTransformer("all-MiniLM-L6-v2")
        logging.info("SentenceTransformer model 'all-MiniLM-L6-v2' is available.")
    except Exception as model_e:
        st.error(f"Failed to load SentenceTransformer model: {model_e}. Please check your internet connection or environment.")
        logging.error(f"SentenceTransformer model loading error: {model_e}")
        st.stop()

    embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}, # Explicitly set device to CPU
        encode_kwargs={'normalize_embeddings': True} # Normalizing embeddings can improve similarity search
    )
    logging.info("HuggingFaceEmbeddings initialized successfully.")

    chroma_db_directory = "db"
    if not os.path.exists(chroma_db_directory):
        st.error(f"ChromaDB directory '{chroma_db_directory}' not found. Please ensure the DB is initialized first (e.g., by running an ingestion script).")
        logging.error(f"ChromaDB directory '{chroma_db_directory}' not found.")
        st.stop()

    logging.info(f"ChromaDB directory '{chroma_db_directory}' found. Attempting to load DB.")
    db = Chroma(persist_directory=chroma_db_directory, embedding_function=embedding)
    logging.info("Chroma DB loaded successfully.")

except Exception as e:
    st.error(f"VectorDB setup error: {e}")
    # Log the full traceback for detailed debugging information.
    logging.exception("Full VectorDB setup traceback:")
    st.stop()

# --- RAG Prompt Template (Modified for dynamic dietary type, goal, and region) ---
# Uses {dietary_type}, {goal}, and {region}.
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
logging.info("RAG Prompt template created with chat history, dynamic dietary type, goal, and region.")

# --- Retrieval QA Chain Setup ---
try:
    # The base QA chain that uses the modified prompt
    # The prompt will be formatted dynamically in the invoke call
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_gemini, # The language model to use for generating answers.
        retriever=db.as_retriever(search_kwargs={"k": 5}), # Increased k to retrieve more documents
        chain_type="stuff", # 'stuff' chain type passes all retrieved documents into the prompt.
        # chain_type_kwargs will be set dynamically in the invoke call
        return_source_documents=True # Include source documents in the output for traceability/debugging.
    )
    logging.info("Retrieval QA Chain initialized successfully.")
except Exception as e:
    st.error(f"QA Chain setup error: {e}")
    logging.exception("Full QA Chain setup traceback:")
    st.stop()

# --- Session History Management ---
# Use a dictionary to store ChatMessageHistory objects for each session ID.
# This store persists across Streamlit reruns within the same process.
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """
    Retrieves or creates a chat message history for a given session ID from the global 'store'.
    Args:
        session_id: The unique identifier for the chat session.
    Returns:
        A ChatMessageHistory object for the session.
    """
    if session_id not in store:
        logging.info(f"Creating new Langchain session history in 'store' for: {session_id}")
        store[session_id] = ChatMessageHistory()
    else:
         logging.info(f"Retrieving existing Langchain session history from 'store' for: {session_id}")
    return store[session_id]

# --- Conversational QA Chain Setup ---
# Wraps the QA chain to automatically manage and pass chat history.
# It uses the get_session_history function and maps input/history keys to the prompt template.
# The dynamic parts of the prompt ({dietary_type}, {goal}, {region}) will be passed in the invoke dictionary.
conversational_qa_chain = RunnableWithMessageHistory(
    qa_chain, # The base QA chain.
    get_session_history, # Function to get session-specific history from 'store'.
    input_messages_key="question", # Key for user's input in the chain (matches diet_prompt).
    history_messages_key="chat_history", # Key for storing chat history within the chain's context (matches diet_prompt).
    # Additional input keys like 'dietary_type', 'goal', and 'region' must be passed in the invoke dictionary
    output_messages_key="answer" # Key for the chain's output.
)
logging.info("Conversational QA Chain initialized with message history.")

# --- Merge Prompt Template (Modified for dynamic dietary type, goal, region, handles table format) ---
# This prompt is used to combine the RAG answer and Groq suggestions.
merge_prompt_template = """
You are a diet planning assistant.
Your goal is to synthesize information from a primary RAG-based answer and several other AI suggestions into a single, coherent, and practical **{dietary_type}** diet plan or suggestion for **{goal}**, tailored for a **{region}** Indian context if a region is specified.
Prioritize the Primary RAG Answer as the core information if it seems relevant and helpful regarding **{dietary_type} {goal}** in an Indian context, especially for the **{region}** if specified. If the RAG answer is generic, error-related, or off-topic, rely more heavily on the Additional Suggestions, ensuring they align with the goal of **{dietary_type} {goal}** and the **{region}** preference.
Incorporate useful and relevant details from the Additional Suggestions if they enhance the practicality and Indian relevance of the advice for **{dietary_type} {goal}** and the **{region}** context.
Ensure the final plan is clear, actionable, and tailored for Indian users, using simple language and common food items.
**IMPORTANT:** If the user's original query or the conversation history contains phrases like "in a table", "table format", "chart", "tabular", or similar requests for structured output, **you MUST present the final diet plan as a clear markdown table.** Otherwise, use a clear list or paragraph format.

If the user's input was *only* a greeting (e.g., "hi", "hello!", "namaste."), respond politely by introducing yourself and asking how you can assist with their diet planning, instead of trying to generate a diet plan based on the information below. For inputs that include a greeting but also contain a query, focus on answering the query while adhering to the goal of **{goal}**, **{dietary_type}**, and **{region}** if previously established in the conversation.

Primary RAG Answer:
{rag}

Additional Suggestions:
- LLaMA Suggestion: {llama}
- Mixtral Suggestion: {mixtral}
- Gemma Suggestion: {gemma}

Refined and Merged {dietary_type} {goal} Diet Plan/Suggestion (Tailored for {region} Indian context, in requested format):
"""
merge_prompt = PromptTemplate.from_template(merge_prompt_template)
logging.info("Merge Prompt template created, includes explicit instructions for table format and dynamic type/goal/region.")


# --- Groq API Integration ---
def groq_diet_answer(model_name: str, query: str, user_diet_preference: str = "any", user_diet_goal: str = "diet", user_region: str = "Indian") -> str:
    """
    Fetches a diet suggestion from a specified Groq model.
    Args:
        model_name: The short name of the Groq model (e.g., "llama", "mixtral").
        query: The user's query.
        user_diet_preference: User's dietary preference ("vegetarian", "non-vegetarian", "vegan", "any").
        user_diet_goal: User's diet goal ("weight gain", "weight loss", "diet").
        user_region: User's regional preference ("South Indian", "North Indian", etc. or "Indian").
    Returns:
        A string containing the diet suggestion or an error message.
    """
    if not GROQ_API_KEY:
        logging.warning(f"Groq API key not available for {model_name}.")
        return f"Groq API key not available."
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        groq_model_map = {
            "llama": "llama3-70b-8192",
            "mixtral": "mixtral-8x7b-32976", # Corrected Mixtral context window size
            "gemma": "gemma2-9b-it"
        }
        actual_model_name = groq_model_map.get(model_name.lower(), model_name)

        # Refined prompt for Groq models - use dynamic dietary preference, goal, and region
        prompt_content = f"User query: '{query}'. Provide a concise, practical **{user_diet_preference}** diet suggestion or food item specifically for **{user_diet_goal}**, tailored for a **{user_region}** Indian context."
        prompt_content += " Focus on readily available ingredients. Be brief."


        payload = {
            "model": actual_model_name,
            "messages": [{"role": "user", "content": prompt_content}],
            "temperature": 0.5,
            "max_tokens": 250
        }
        logging.info(f"Calling Groq API with model: {actual_model_name} for query: '{query}' ({user_diet_preference} {user_diet_goal}, {user_region})")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data and data.get('choices') and len(data['choices']) > 0 and data['choices'][0].get('message'):
            return data['choices'][0]['message']['content']
        else:
            logging.warning(f"Groq API for {actual_model_name} returned an unexpected or empty response: {data}")
            return f"No suggestion from {actual_model_name} (empty or malformed response)."
    except requests.exceptions.Timeout:
        logging.error(f"Timeout error from {model_name} ({actual_model_name}) for query: '{query}'")
        return f"Timeout error from {model_name}."
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error from {model_name} ({actual_model_name}): {e}")
        return f"Request error from {model_name}: {e}"
    except Exception as e:
        logging.error(f"Unexpected error from {model_name} ({actual_model_name}): {e}")
        return f"Error from {model_name}: {e}"

# Use Streamlit's caching to avoid repeated Groq calls for the same query and preference/goal/region
@st.cache_data(show_spinner=False, ttl=3600) # Cache for 1 hour
def cached_groq_answers(query: str, diet_preference: str, diet_goal: str, region: str) -> dict:
    """
    Fetches and caches suggestions from multiple Groq models concurrently.
    Uses diet_preference, diet_goal, and region in the cache key and passes them to groq_diet_answer.
    """
    logging.info(f"Fetching cached Groq answers for query: '{query}', preference: '{diet_preference}', goal: '{diet_goal}', region: '{region}'")
    models = ["llama", "mixtral", "gemma"]
    results = {}
    if not GROQ_API_KEY:
        logging.warning("Groq API key not set, skipping Groq calls.")
        return {k: "Groq API key not available." for k in models}

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        # Pass the detected diet_preference, diet_goal, and region to the groq_diet_answer function
        future_to_model = {executor.submit(groq_diet_answer, name, query, diet_preference, diet_goal, region): name for name in models}
        for future in future_to_model:
            model_name = future_to_model[future]
            try:
                results[model_name] = future.result()
                logging.info(f"Successfully fetched {model_name} suggestion.")
            except Exception as e:
                logging.error(f"Error fetching {model_name} suggestion in thread: {e}")
                results[model_name] = f"Failed to get suggestion from {model_name}: {e}"
    return results

# --- Greeting, Preference, Goal, and Region Helpers ---
GREETINGS = ["hi", "hello", "hey", "namaste", "yo", "vanakkam", "bonjour", "salaam", "good morning", "good afternoon", "good evening"]

def is_greeting(query: str) -> bool:
    """Checks if the query is a simple greeting."""
    if not query:
        return False
    # Remove punctuation and check against greetings list
    cleaned_query = query.translate(str.maketrans('', '', string.punctuation)).strip().lower()
    return cleaned_query in GREETINGS

def extract_diet_preference(query: str) -> str:
    """Extracts diet preference (vegetarian, non-vegetarian, vegan, any) from the query."""
    query_lower = query.lower()
    if "non-veg" in query_lower or "non veg" in query_lower or "nonvegetarian" in query_lower or "non vegetarian" in query_lower:
        return "non-vegetarian"
    elif "vegan" in query_lower:
        return "vegan"
    elif "veg" in query_lower or "vegetarian" in query_lower:
        return "vegetarian"
    return "any" # Default to 'any' if no specific preference is found

def extract_diet_goal(query: str) -> str:
    """Extracts diet goal (weight gain, weight loss, diet) from the query."""
    query_lower = query.lower()
    if "weight gain" in query_lower or "gain weight" in query_lower or "gain" in query_lower:
        return "weight gain"
    elif "weight loss" in query_lower or "loss weight" in query_lower or "loss" in query_lower:
        return "weight loss"
    return "diet" # Default to 'diet' if no specific goal is mentioned

def extract_regional_preference(query: str) -> str:
    """Extracts regional preference (e.g., South Indian, North Indian) from the query."""
    query_lower = query.lower()
    # Use regex to find common regional terms
    match = re.search(r"(south|north|west|east) india(?:n)?|bengali|punjabi|maharashtrian|gujarati|tamil|kannada|telugu|malayalam|kanyakumari", query_lower)
    if match:
        # Capitalize the first letter of each word for better formatting
        return " ".join([word.capitalize() for word in match.group(0).replace('indian', 'Indian').split()])
    return "Indian" # Default to 'Indian' if no specific region is mentioned

def contains_table_request(query: str) -> bool:
    """Checks if the query contains keywords requesting a table format."""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in ["table", "tabular", "chart", "in a table", "in table format", "as a table"])

# --- Streamlit UI ---
st.set_page_config(page_title="🍱 Indian Diet Advisor", layout="wide")

with st.sidebar:
    st.markdown("👤 **Created by Lord d'Artagnan**")
    st.markdown("---")
    use_llms_toggle = st.toggle("🔄 Include expanded suggestions from other models", value=True, help="Fetch additional suggestions from LLaMA, Mixtral, and Gemma via Groq API.")

st.title("🥗 Personalized Indian Diet Recommendation")
st.markdown("Ask anything related to Indian diet and health. Suggestions are tailored for Indian users.")

# --- Session ID and History Management ---
# Initialize session_id in Streamlit's session state if it doesn't exist.
if 'session_id' not in st.session_state:
    st.session_state.session_id = "session_" + os.urandom(8).hex()
    st.session_state.messages = [] # Initialize UI messages for a brand new session
    st.session_state.dietary_type = "any" # Initialize default dietary type
    st.session_state.diet_goal = "diet" # Initialize default diet goal
    st.session_state.region = "Indian" # Initialize default region
    st.session_state.table_format_requested = False # Initialize table format flag
    # Ensure Langchain history is created in 'store' for the initial session ID
    get_session_history(st.session_state.session_id)
    logging.info(f"New Streamlit session started. Initial session_id: {st.session_state.session_id}")
else:
    # Ensure the Langchain history is loaded into 'store' if the session state already has an ID
    # This handles cases where the app reruns but the session state persists.
    get_session_history(st.session_state.session_id)
    # Ensure dietary_type, diet_goal, region, and table_format_requested are in session state on rerun
    if 'dietary_type' not in st.session_state:
        st.session_state.dietary_type = "any"
    if 'diet_goal' not in st.session_state:
        st.session_state.diet_goal = "diet"
    if 'region' not in st.session_state:
         st.session_state.region = "Indian"
    if 'table_format_requested' not in st.session_state:
         st.session_state.table_format_requested = False


    logging.info(f"Streamlit session state has existing session_id: {st.session_state.session_id}. Langchain history loaded. Current type: {st.session_state.dietary_type}, Goal: {st.session_state.diet_goal}, Region: {st.session_state.region}, Table Requested: {st.session_state.table_format_requested}")


# Allow users to input or change the session ID.
session_id_input = st.text_input(
    "Session ID:",
    value=st.session_state.session_id,
    help="Use a unique ID for each conversation. Changing this will start a new conversation or load an existing one."
)

# Logic for handling session ID changes by the user.
if session_id_input and session_id_input != st.session_state.session_id:
    old_session_id = st.session_state.session_id
    st.session_state.session_id = session_id_input
    logging.info(f"User changed session ID from {old_session_id} to: {st.session_state.session_id}")

    # Retrieve the Langchain history for the (potentially existing) new session ID.
    current_langchain_history = get_session_history(st.session_state.session_id)
    logging.info(f"Retrieved Langchain history from 'store' for session {st.session_state.session_id}. Message count: {len(current_langchain_history.messages)}")

    # Rebuild st.session_state.messages (for UI display) from the retrieved Langchain history.
    new_ui_messages = []
    # Reset dietary type, goal, region, and table format flag when switching sessions
    st.session_state.dietary_type = "any"
    st.session_state.diet_goal = "diet"
    st.session_state.region = "Indian" # Reset region
    st.session_state.table_format_requested = False # Reset table format flag


    for message_obj in current_langchain_history.messages:
        if isinstance(message_obj, HumanMessage): # Check type using isinstance
            new_ui_messages.append({"role": "user", "content": message_obj.content})
            # Attempt to update type/goal/region/table_format from historical user messages (optional but helpful)
            st.session_state.dietary_type = extract_diet_preference(message_obj.content) or st.session_state.dietary_type
            st.session_state.diet_goal = extract_diet_goal(message_obj.content) or st.session_state.diet_goal
            st.session_state.region = extract_regional_preference(message_obj.content) or st.session_state.region
            if contains_table_request(message_obj.content):
                 st.session_state.table_format_requested = True


        elif isinstance(message_obj, AIMessage): # Check type using isinstance
            new_ui_messages.append({"role": "assistant", "content": message_obj.content})
        else:
            logging.warning(f"Encountered unknown message type in Langchain history: {type(message_obj)}")

    st.session_state.messages = new_ui_messages # Update Streamlit's UI message list
    logging.info(f"UI messages (st.session_state.messages) updated from Langchain history. New UI message count: {len(st.session_state.messages)}")
    logging.info(f"Inferred type/goal/region/table_format after loading history: {st.session_state.dietary_type}/{st.session_state.diet_goal}/{st.session_state.region}/{st.session_state.table_format_requested}")

    st.toast(f"Switched to session: {st.session_state.session_id}. History loaded.")
    st.experimental_rerun() # Use experimental_rerun for session state changes triggering UI update.

# Ensure messages list exists even if session state was just initialized or cleared
if "messages" not in st.session_state:
    st.session_state.messages = []
    logging.warning("'messages' not found in st.session_state after session_id handling. Re-initializing to empty list.")


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
query = st.chat_input("Enter your diet-related question:", key="user_query_input")

if query:
    logging.info(f"Input received: '{query}' for session: {st.session_state.session_id}")

    # Add user message to Streamlit's state for display
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    # Check if the query is a pure greeting
    if is_greeting(query):
        logging.info("Query identified as a pure greeting.")
        # Generalized greeting
        greeting_response = "Namaste! I am your diet planning assistant. How may I help you with your Indian diet and health questions today?"
        st.session_state.messages.append({"role": "assistant", "content": greeting_response})
        with st.chat_message("assistant"):
            st.markdown(greeting_response)

        # Add the greeting response to Langchain history as an AI message
        get_session_history(st.session_state.session_id).add_ai_message(greeting_response)
        logging.info("Greeting response added to Langchain history.")

    else:
        logging.info("Query is NOT a pure greeting. Proceeding with RAG/Merge.")
        # Display assistant thinking message
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                rag_answer = ""
                rag_success = False # Flag to track if RAG was successful

                # Extract dietary type, goal, and region from the current query
                current_dietary_type = extract_diet_preference(query)
                current_diet_goal = extract_diet_goal(query)
                current_region = extract_regional_preference(query)
                current_table_request = contains_table_request(query)


                # Update session state if a preference, goal, region, or table request is detected in the current query
                if current_dietary_type != "any":
                    st.session_state.dietary_type = current_dietary_type
                    logging.info(f"Updated session dietary type to: {st.session_state.dietary_type}")
                if current_diet_goal != "diet":
                    st.session_state.diet_goal = current_diet_goal
                    logging.info(f"Updated session diet goal to: {st.session_state.diet_goal}")
                if current_region != "Indian":
                    st.session_state.region = current_region
                    logging.info(f"Updated session region to: {st.session_state.region}")
                if current_table_request:
                     st.session_state.table_format_requested = True
                     logging.info("Updated session table_format_requested to: True")


                logging.info(f"Using session dietary type: {st.session_state.dietary_type}, goal: {st.session_state.diet_goal}, region: {st.session_state.region}, table requested: {st.session_state.table_format_requested} for RAG and Groq.")

                try:
                    logging.info(f"Invoking conversational_qa_chain for session: {st.session_state.session_id} with type: {st.session_state.dietary_type}, goal: {st.session_state.diet_goal}, region: {st.session_state.region}")
                    # Invoke the conversational chain which handles RAG and history
                    # Pass the detected dietary type, goal, and region to format the RAG prompt
                    result = conversational_qa_chain.invoke(
                        {"question": query,
                         "dietary_type": st.session_state.dietary_type,
                         "goal": st.session_state.diet_goal,
                         "region": st.session_state.region},
                        config={"configurable": {"session_id": st.session_state.session_id}}
                    )

                    # Extract the answer from the result. The structure depends on the chain type.
                    # For RetrievalQA with return_source_documents=True, the answer is usually in 'answer'.
                    if isinstance(result, dict) and "answer" in result:
                         rag_answer = result["answer"]
                         rag_success = True
                         logging.info(f"RAG answer obtained successfully: '{rag_answer[:100]}...'")
                         # Optional: Log source documents if available
                         if "source_documents" in result:
                             logging.info(f"Retrieved {len(result['source_documents'])} source documents.")
                             # Log snippets of source documents for debugging
                             for i, doc in enumerate(result['source_documents']):
                                 logging.info(f"Source {i+1}: {doc.page_content[:200]}... (Source: {doc.metadata.get('source', 'N/A')})")

                    elif isinstance(result, str):
                         rag_answer = result # Handle cases where the chain might return a string directly
                         rag_success = True # Consider it a success if a string is returned, though unexpected format
                         logging.warning("RAG chain returned a string directly, expected a dict with 'answer'.")
                    else:
                         rag_answer = "Could not retrieve a specific answer from the knowledge base at this time (unexpected result format)."
                         logging.error(f"RAG chain returned unexpected result format: {type(result)}, content: {result}")

                except Exception as e:
                    rag_answer = f"Sorry, I encountered an error while consulting my knowledge base: {e}"
                    logging.exception("Error during RAG chain invocation:")
                    rag_success = False # RAG failed

                # Get suggestions from other models concurrently if enabled
                # Pass the current session's dietary_type, diet_goal, and region to the cached_groq_answers function
                user_preference_for_groq = st.session_state.dietary_type
                user_goal_for_groq = st.session_state.diet_goal
                user_region_for_groq = st.session_state.region

                logging.info(f"Fetching Groq suggestions for preference: {user_preference_for_groq}, goal: {user_goal_for_groq}, region: {user_region_for_groq}")

                llama_suggestion = mixtral_suggestion = gemma_suggestion = ""
                if use_llms_toggle:
                    if GROQ_API_KEY:
                        # Use cached function for Groq calls, passing the detected preference, goal, and region
                        groq_results = cached_groq_answers(query, user_preference_for_groq, user_goal_for_groq, user_region_for_groq)
                        llama_suggestion = groq_results.get("llama", "N/A")
                        mixtral_suggestion = groq_results.get("mixtral", "N/A")
                        gemma_suggestion = groq_results.get("gemma", "N/A")
                        logging.info(f"Groq suggestions fetched. Llama: {llama_suggestion[:50]}..., Mixtral: {mixtral_suggestion[:50]}..., Gemma: {gemma_suggestion[:50]}...")
                    else:
                        llama_suggestion = mixtral_suggestion = gemma_suggestion = "Groq API key not available."
                        logging.warning("Groq API key not available, skipping Groq calls.")
                else:
                    llama_suggestion = mixtral_suggestion = gemma_suggestion = "Suggestions from other models are disabled."
                    logging.info("Suggestions from other models are disabled.")

                # Merge the RAG answer and Groq suggestions using Gemini
                final_merged_response = ""
                try:
                    # The merge prompt template itself now contains logic for table format and uses dynamic type/goal/region
                    final_prompt_text = merge_prompt.format(
                        rag=rag_answer,
                        llama=llama_suggestion,
                        mixtral=mixtral_suggestion,
                        gemma=gemma_suggestion,
                        dietary_type=st.session_state.dietary_type, # Pass the session's detected type to the merge prompt
                        goal=st.session_state.diet_goal, # Pass the session's detected goal to the merge prompt
                        region=st.session_state.region # Pass the session's detected region to the merge prompt
                    )

                    logging.info("Invoking Gemini for final merge.")
                    # Use llm_gemini.invoke() which is suitable for single turn prompts
                    final_merged_response = llm_gemini.invoke(final_prompt_text)

                    # Ensure the response is a string
                    if hasattr(final_merged_response, 'text'):
                         final_merged_response = final_merged_response.text
                    elif not isinstance(final_merged_response, str):
                         final_merged_response = str(final_merged_response)

                    logging.info("Final response generated after merging.")

                except Exception as e:
                    final_merged_response = f"Sorry, I encountered an error while trying to put all the information together: {e}"
                    logging.exception("Error during final merge invocation with Gemini:")

                # Add final response to Streamlit's state and display
                st.session_state.messages.append({"role": "assistant", "content": final_merged_response})
                st.markdown(final_merged_response)

                # Add the final response to Langchain history as an AI message
                get_session_history(st.session_state.session_id).add_ai_message(final_merged_response)
                logging.info("Final response added to Langchain history.")


st.markdown("---")
st.markdown("Disclaimer: This advisor provides general suggestions based on AI models and a knowledge base. Consult a qualified healthcare professional or dietitian for personalized medical or dietary advice.")

logging.info("Application finished processing request.")
