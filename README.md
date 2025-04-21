
<p align="center">
  <img src="assets/logo.png" alt="Diet-Suggest Logo" width="200"/>
</p>

<h1 align="center">🥗 Diet-Suggest — Your AI-Powered Dietitian</h1>

<p align="center"><em>Eat smart, live strong — backed by AI.</em></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" height="32"/>
  <img src="https://img.shields.io/badge/Streamlit-%E2%9D%A4-red?logo=streamlit" height="32"/>
  <img src="https://img.shields.io/badge/License-MIT-green?logo=open-source-initiative" height="32"/>
  <img src="https://img.shields.io/badge/Gemini-Powered-blueviolet?logo=google" height="32"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" height="32"/>
  <img src="https://img.shields.io/badge/LangChain-Enabled-yellowgreen?logo=chainlink" height="32"/>
  <img src="https://img.shields.io/badge/Groq-Compatible-orange?logo=lightning" height="32"/>
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20DB-critical?logo=databricks" height="32"/>
</p>

---

## 🌟 Key Features

- 🧠 **LLM-Driven Diet Plans**  
  Personalized diet suggestions using Gemini with RAG (Retrieval-Augmented Generation).

- ❤️ **Health-Aware Recommendations**  
  Handles diabetes, hypertension, weight goals, and other medical conditions.

- 📍 **Location-Aware Output**  
  Recommends diet plans relevant to your city or region.

- 🗂 **Embedded FAQ Matching**  
  Uses vector embeddings to intelligently match queries to diet facts.

- 🐳 **Docker & DevContainer Support**  
  Works seamlessly in isolated environments.

- ⚡ **Groq-Compatible**  
  Optimized for future high-speed inference.

---

## 🧠 Architecture Overview

<p align="center">
  <img src="assets/architecture.png" alt="App Architecture" width="600"/>
</p>

---

## 📁 Directory Structure

```
Diet-Suggest/
├── streamlit_app.py         # 🎛️ Main Streamlit interface
├── build_vector_db.py       # 🧠 Builds the Vector DB (FAQs, Diet KB)
├── requirements.txt         # 📦 Python dependencies
├── Dockerfile               # 🐳 Docker container config
├── .devcontainer/           # ⚙️ Dev container (VS Code)
├── assets/                  # 🖼️ Logos and diagrams
└── README.md                # 📖 This file
```

---

## ⚙️ Setup Guide

### 🔁 1. Clone the Repository

```bash
git clone https://github.com/DYNOSuprovo/Diet-Suggest.git
cd Diet-Suggest
```

### 📦 2. Install Dependencies

Make sure you have **Python 3.9+** installed.

```bash
pip install -r requirements.txt
```

### 🔑 3. Configure API Keys

Create a `.env` file in the root directory and add your keys:

```env
GEMINI_API_KEY=your_gemini_api_key_here
# Optional:
# HUGGINGFACE_API_KEY=your_hf_key
# OPENAI_API_KEY=your_openai_key
# GROQ_API_KEY=your_groq_key
```

### 🧠 4. Build the Vector Database

```bash
python build_vector_db.py
```

### ▶️ 5. Launch the Streamlit App

```bash
streamlit run streamlit_app.py
```

You can now:
- Select medical conditions
- Enter your city
- Ask diet-related questions
- Get instant AI-powered suggestions 💡

---

## 🐳 Docker Instructions

### 🔧 Build & Run with Docker

Ensure Docker is installed, then:

```bash
docker build -t diet-suggest .
docker run -p 8501:8501 diet-suggest
```

Visit the app at: [http://localhost:8501](http://localhost:8501)

---

## 📸 Screenshots

_Include one or more screenshots of the app UI here for better context._

```
assets/preview.png
```

---

## 🤝 Contribution

Contributions are welcome!  
If you have new ideas or major changes, open an issue first to discuss.

---

## 📜 License

Licensed under the [MIT License](LICENSE).

---

## 🔗 Useful Links

- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain](https://docs.langchain.com/)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [ChromaDB](https://www.trychroma.com/)
- [Gemini API](https://ai.google.dev/)
- [Groq](https://groq.com/)

---

<p align="center"><em>Crafted with 💡 by Suprovo (a.k.a. LDrago) — Empowering healthy lives with AI.</em></p>
