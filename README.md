# 🥗 Diet-Suggest — Your AI-Powered Dietitian

**Diet-Suggest** is an AI-based food recommendation app built using **Streamlit**, **LangChain**, and **Gemini**. It delivers personalized diet plans tailored to your medical conditions, preferences, and location — using smart retrieval and powerful LLMs.

> 🚀 *"Eat smart, live strong — backed by AI."*

---

## 🌟 Key Features

- 🧠 **LLM-Driven Diet Plans**  
  Get real-time diet suggestions via Gemini using RAG (Retrieval-Augmented Generation).

- ❤️ **Health-Aware Recommendations**  
  Adaptable for conditions like diabetes, high BP, weight goals, and more.

- 📍 **Location-Aware Output**  
  Offers dietary suggestions based on your city/region.

- 🗂 **Embedded FAQ Matching**  
  Built-in embeddings to smartly match your queries with pre-trained diet info.

- 🐳 **Docker & DevContainer Support**  
  Run it anywhere — even in isolated environments.

---

## 🧠 Architecture Overview

```mermaid
flowchart TD
    A[User Input] --> B[Query Processor]
    B --> C{Medical\nConditions?}
    C --> D[Vector DB (FAISS/Chroma)]
    D --> E[LangChain + Gemini]
    E --> F[Smart Response]
    F --> G[Streamlit UI]
```

---

## 📁 Directory Structure

```
Diet-Suggest/
├── streamlit_app.py          # 🎛️ Main Streamlit interface
├── build_vector_db.py        # 🧠 Vector DB builder for FAQs
├── requirements.txt          # 📦 Python dependencies
├── Dockerfile                # 🐳 Docker container setup
├── .devcontainer/            # ⚙️ Dev container for VS Code
└── README.md                 # 📖 You're here!
```

---

## ⚙️ Setup Guide

### 🔁 1. Clone the Repo

```bash
git clone https://github.com/DYNOSuprovo/Diet-Suggest.git
cd Diet-Suggest
```

### 📦 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔑 3. Add Your API Keys

Create a `.env` file:

```env
GEMINI_API_KEY=your_key_here
```

(Optional: Add HuggingFace/OpenAI keys if used)

---

### 🧠 4. Build Vector DB (FAQ + Diet Knowledge)

```bash
python build_vector_db.py
```

---

### ▶️ 5. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

You’ll be able to:
- Select medical conditions
- Choose location
- Ask diet-related questions
- Get instant AI-driven suggestions 💡

---

## 🐳 Docker Instructions

### 🔧 Build & Run Locally

```bash
docker build -t diet-suggest .
docker run -p 8501:8501 diet-suggest
```

> App will be available at `http://localhost:8501`

---

## 📸 Screenshots (Optional)

> *(Add screenshots of Streamlit interface once finalized)*  
> Example:
> ![Homepage Preview](assets/preview.png)

---

## 🤝 Contribution

Pull requests and suggestions are welcome!  
If you're proposing major changes, kindly open an issue to discuss it first.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🔗 Useful Links

- [Streamlit Docs](https://docs.streamlit.io/)
- [LangChain](https://docs.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Gemini API](https://cloud.google.com/vertex-ai/docs/generative-ai/overview)

---

*Made with 💡 by Suprovo (LDrago) — powering healthy lives with AI.*
