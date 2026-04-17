# 📄 Research Paper Q&A Agent — Capstone Project

A conversational agent that answers questions about AI/ML research papers using:
- **Google Gemini 2.5 Flash** as the LLM
- **LangGraph** for the agentic state machine (7 nodes)
- **ChromaDB** for vector-based RAG (12 documents)
- **DuckDuckGo Search** as the web tool
- **Streamlit** for the chat UI

---

## 🗂️ Project Files

```
research_paper_qa/
├── agent.py                   ← Core agent module (LangGraph + ChromaDB)
├── app.py                     ← Streamlit UI (run this to launch the app)
├── day13_capstone.ipynb       ← Completed Jupyter notebook
├── generate_documentation.py  ← Generates the submission PDF
├── requirements.txt           ← Python dependencies
├── .env                       ← API keys (YOU MUST FILL THIS IN)
├── .gitignore                 ← Excludes .env and __pycache__
└── README.md                  ← This file
```

---

## ⚙️ Setup: Step-by-Step

### STEP 1 — Get Your Gemini API Key (Free)
1. Go to: https://aistudio.google.com/app/apikey
2. Click **"Create API key"**
3. Copy the key (it starts with `AIza...`)

### STEP 2 — Fill in Your .env File
Open `.env` in any text editor and replace:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```
with:
```
GOOGLE_API_KEY=AIzaSy...your_actual_key...
```


### STEP 3 — Create a Virtual Environment (Recommended)
```bash
# Navigate to the project folder
cd research_paper_qa

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
```

### STEP 4 — Install Dependencies
```bash
pip install -r requirements.txt
```
This will take 2-5 minutes. It downloads sentence-transformers (~90MB) and other packages.

### STEP 5 — Run the Streamlit App
```bash
streamlit run app.py
```
The app opens automatically at `http://localhost:8501`.

### STEP 6 — Run the Jupyter Notebook
```bash
jupyter notebook day13_capstone.ipynb
```
Then run all cells top to bottom (Kernel → Restart & Run All).

### STEP 7 — Generate the Submission PDF
Open `generate_documentation.py`, fill in your Name/Roll/Batch at the top, then:
```bash
python generate_documentation.py
```
This creates `Project_Documentation.pdf`.

---

## 🔧 How to Modify

| What you want to change | Where to change it |
|---|---|
| LLM model name | `agent.py` → `LLM_MODEL = "gemini-2.5-flash"` |
| Add more KB documents | `agent.py` → `DOCUMENTS` list |
| Change retrieval count (top-k) | `agent.py` → `retrieval_node` → `n_results=3` |
| Change faithfulness threshold | `agent.py` → `FAITHFULNESS_THRESHOLD = 0.7` |
| Change tool (e.g. use arXiv API) | `agent.py` → `tool_node` function |
| Change system prompt | `agent.py` → `answer_node` → `system_content` |
| Change UI title/description | `app.py` → top section |

---

## 📦 Submission Checklist

- [ ] `.env` filled in with your Gemini API key
- [ ] `day13_capstone.ipynb` — all cells run without errors
- [ ] Part 8 Summary filled in (name, roll number, batch, RAGAS scores)
- [ ] `app.py` launches and chat works
- [ ] Conversation memory tested (3+ follow-up questions)
- [ ] `Project_Documentation.pdf` generated with screenshots added
- [ ] All files zipped into ONE ZIP folder
- [ ] GitHub repo created and link ready
- [ ] Google Form submitted before April 21, 2026 11:59 PM

---

## ❓ Troubleshooting

**`GOOGLE_API_KEY not set` error:**
→ Open `.env` and make sure the key is pasted correctly, no extra spaces.

**`ModuleNotFoundError`:**
→ Make sure your virtual environment is activated and you ran `pip install -r requirements.txt`.

**Slow first run:**
→ sentence-transformers downloads the embedding model (~90MB) on first run. Wait for it.

**`streamlit: command not found`:**
→ Activate your virtual environment first: `source venv/bin/activate`

**ChromaDB error on Colab:**
→ Use `chromadb.Client()` (in-memory). Do NOT use `chromadb.PersistentClient` on Colab.
