"""
agent.py — Research Paper Q&A Agent (Core Module)
==================================================
Shared module imported by both the notebook and the Streamlit app.

Uses:
  • LangGraph StateGraph (7 nodes)
  • ChromaDB for RAG
  • Google Gemini (gemini-2.5-flash) as the LLM
  • sentence-transformers for local embeddings (no API key needed)
  • DuckDuckGo web search as the extra tool
  • MemorySaver for conversation memory

MODIFY HERE if you want to:
  - Change the LLM model name
  - Change the number of retrieved chunks (n_results)
  - Change the faithfulness threshold
  - Add your own PDF documents
"""

import os
import re
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── 🔑 LLM Initialisation ─────────────────────────────────────────────────────
# MODIFY: Change model name here if needed.
# Available Gemini models: gemini-2.5-flash, gemini-1.5-pro, gemini-1.5-flash
LLM_MODEL = "gemini-2.5-flash"

def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError(
            "❌ GOOGLE_API_KEY not set!\n"
            "Open .env and paste your Gemini API key.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0,
        google_api_key=api_key
    )

# ── 📚 Knowledge Base Documents ───────────────────────────────────────────────
# These are 12 research-paper-domain documents covering key AI/ML topics.
# MODIFY: Replace or add more documents to expand the knowledge base.
# Each document needs: id, topic, text (100-400 words recommended).

DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Transformer Architecture",
        "text": """The Transformer architecture, introduced in the paper 'Attention Is All You Need' (Vaswani et al., 2017),
        revolutionised natural language processing by replacing recurrent neural networks with self-attention mechanisms.
        The model consists of an encoder and a decoder, each made up of multiple identical layers. The encoder maps an
        input sequence to a sequence of continuous representations, while the decoder generates the output sequence
        one element at a time. The key innovation is the multi-head self-attention mechanism, which allows the model
        to simultaneously attend to information from different representation subspaces at different positions.
        Positional encodings are added to input embeddings to inject information about the position of tokens in the
        sequence. The Transformer uses three types of attention: encoder self-attention, decoder self-attention, and
        encoder-decoder attention. Feed-forward networks are applied independently to each position. Layer normalisation
        and residual connections are used throughout. The architecture achieves state-of-the-art performance on machine
        translation tasks and became the foundation for models like BERT and GPT."""
    },
    {
        "id": "doc_002",
        "topic": "BERT — Bidirectional Transformers",
        "text": """BERT (Bidirectional Encoder Representations from Transformers), introduced by Devlin et al. in 2018,
        is a pre-training approach for NLP that uses the Transformer encoder. Unlike previous models that read text
        sequentially, BERT reads the entire sequence of words at once, making it deeply bidirectional. It is pre-trained
        on two tasks: Masked Language Modelling (MLM), where 15% of input tokens are masked and the model learns to
        predict them; and Next Sentence Prediction (NSP), where the model learns to understand sentence relationships.
        BERT is fine-tuned on downstream tasks by adding a simple output layer. It achieved state-of-the-art results
        on eleven NLP tasks including question answering, natural language inference, and named entity recognition.
        The base model has 110 million parameters and the large model has 340 million. BERT's bidirectional context
        understanding was a significant advancement over unidirectional models like GPT. It uses WordPiece tokenisation
        and is trained on BooksCorpus and English Wikipedia."""
    },
    {
        "id": "doc_003",
        "topic": "GPT and Autoregressive Language Models",
        "text": """The GPT (Generative Pre-trained Transformer) series, developed by OpenAI, are autoregressive language
        models that generate text by predicting the next token given all previous tokens. GPT-1 (2018) demonstrated
        that language models pre-trained on large text corpora can be fine-tuned for downstream tasks with minimal
        labelled data. GPT-2 (2019) scaled this approach and showed emergent capabilities in text generation, translation,
        and summarisation without task-specific training. GPT-3 (2020) with 175 billion parameters demonstrated few-shot
        and zero-shot learning capabilities across a wide range of tasks using only natural language prompts. GPT-4 (2023)
        is multimodal and outperforms GPT-3 on most benchmarks. These models use a decoder-only Transformer architecture.
        Training involves predicting the next token in a sequence using cross-entropy loss. In-context learning, where
        the model learns from examples provided in the prompt without updating weights, emerged as a surprising property
        of large GPT models. The models are trained on diverse internet text using self-supervised learning."""
    },
    {
        "id": "doc_004",
        "topic": "Retrieval-Augmented Generation (RAG)",
        "text": """Retrieval-Augmented Generation (RAG), introduced by Lewis et al. in 2020, combines parametric memory
        (model weights) with non-parametric memory (external document retrieval) to improve knowledge-intensive NLP tasks.
        In RAG, for each input, relevant documents are retrieved from a large corpus using a dense retrieval model like
        DPR (Dense Passage Retrieval), and then the input and retrieved documents are fed to a seq2seq model (like BART)
        to generate the output. There are two variants: RAG-Sequence uses the same retrieved document for the entire
        output, while RAG-Token can retrieve different documents for each output token. RAG outperforms parametric
        seq2seq models and task-specific retrieval-and-read models on open-domain question answering, abstractive
        question answering, and fact verification tasks. The approach reduces hallucinations because answers must
        be grounded in retrieved evidence. Modern RAG pipelines use vector databases like ChromaDB or FAISS for
        efficient similarity search and large language models as the reader. RAG is particularly valuable when
        factual accuracy and up-to-date information are important."""
    },
    {
        "id": "doc_005",
        "topic": "Attention Mechanisms",
        "text": """Attention mechanisms allow neural networks to focus on relevant parts of the input when generating
        each part of the output. Bahdanau et al. (2015) introduced the first attention mechanism for neural machine
        translation, enabling the decoder to look back at all encoder hidden states rather than just the final one.
        Scaled dot-product attention, used in Transformers, computes attention scores as: Attention(Q, K, V) =
        softmax(QK^T / sqrt(d_k)) * V, where Q (queries), K (keys), and V (values) are linear projections of the
        input. Multi-head attention runs several attention functions in parallel and concatenates the results,
        allowing the model to attend to information from different representation subspaces. Self-attention relates
        different positions of a single sequence to compute representations of that sequence. Cross-attention relates
        positions in one sequence (decoder) to positions in another (encoder). Sparse attention variants like
        Longformer and BigBird reduce the O(n²) complexity of full self-attention to handle longer sequences.
        Flash Attention is an IO-aware exact attention algorithm that uses tiling to reduce memory reads/writes."""
    },
    {
        "id": "doc_006",
        "topic": "Diffusion Models for Image Generation",
        "text": """Diffusion models are a class of generative models that learn to reverse a gradual noising process.
        During training, Gaussian noise is progressively added to data over T timesteps (forward process). The model
        learns to reverse this process by predicting and removing the noise at each step (reverse process). Denoising
        Diffusion Probabilistic Models (DDPM), introduced by Ho et al. (2020), formulated this as a Markov chain and
        showed high-quality image generation. Stable Diffusion uses a latent diffusion model that operates in a
        compressed latent space rather than pixel space, making it computationally efficient. DALL-E 2 and Imagen
        condition the diffusion process on text embeddings using classifier-free guidance to generate images from
        text descriptions. Score matching and score-based generative models provide an alternative but equivalent
        mathematical framework. Diffusion models outperform GANs on image quality metrics and offer better mode
        coverage. Applications include image synthesis, inpainting, super-resolution, audio generation, and
        protein structure prediction. The key trade-off is slow sampling speed compared to single-forward-pass
        generative models like VAEs and GANs."""
    },
    {
        "id": "doc_007",
        "topic": "Reinforcement Learning from Human Feedback (RLHF)",
        "text": """Reinforcement Learning from Human Feedback (RLHF) is a technique to align language models with
        human preferences. The process has three steps. First, a supervised fine-tuning (SFT) model is trained on
        high-quality demonstration data. Second, a reward model is trained on human preference comparisons — given
        two model outputs, humans rank which is better, and the reward model learns to predict these preferences.
        Third, the SFT model is fine-tuned using proximal policy optimisation (PPO) to maximise the reward model's
        score while staying close to the original model via a KL divergence penalty. This approach was used to
        create InstructGPT and ChatGPT, which are significantly preferred by humans over models trained with
        standard language modelling. Challenges include reward hacking (where the model exploits reward model
        weaknesses), the cost of human labelling, and scalable oversight (aligning models smarter than human
        evaluators). Direct Preference Optimisation (DPO) is a simpler alternative that bypasses the reward model
        and directly optimises the policy using preference data with a classification loss."""
    },
    {
        "id": "doc_008",
        "topic": "Graph Neural Networks (GNNs)",
        "text": """Graph Neural Networks (GNNs) are deep learning models designed to operate on graph-structured data,
        where relationships between entities are explicitly modelled as edges. The core operation is message passing:
        each node aggregates feature information from its neighbours, applies a learnable transformation, and updates
        its own representation. This is repeated for multiple layers, allowing nodes to gather information from
        multi-hop neighbourhoods. Graph Convolutional Networks (GCN) by Kipf and Welling (2017) use a spectral
        convolution simplified to first-order neighbourhood aggregation. GraphSAGE samples and aggregates features
        from local neighbourhoods, enabling inductive learning on unseen nodes. Graph Attention Networks (GAT) use
        attention coefficients to weight neighbour contributions. Applications of GNNs include node classification,
        link prediction, graph classification, molecular property prediction in drug discovery, social network analysis,
        knowledge graph completion, recommendation systems, and traffic forecasting. A major challenge is over-smoothing,
        where node representations become indistinguishable as the number of layers increases. Heterogeneous GNNs
        handle graphs with multiple node and edge types."""
    },
    {
        "id": "doc_009",
        "topic": "Federated Learning",
        "text": """Federated Learning (FL), introduced by McMahan et al. in 2017, is a machine learning paradigm that
        trains models across multiple decentralised devices or servers holding local data samples, without sharing
        raw data. Instead of uploading data to a central server, each participating device trains a local model on
        its own data and sends only model updates (gradients or weights) to a central aggregator. The aggregator
        combines these updates using FedAvg (Federated Averaging), which computes a weighted average of local models.
        This approach preserves data privacy and is useful in healthcare (patient records), mobile devices (keyboard
        predictions), and financial services (fraud detection). Key challenges include statistical heterogeneity
        (non-IID data across clients), system heterogeneity (varying compute and communication capacity), and
        privacy (differential privacy or secure aggregation are used to protect against inference attacks on
        gradients). Communication efficiency is improved using gradient compression and local SGD steps.
        Cross-device FL involves millions of mobile devices, while cross-silo FL involves a small number of
        institutional partners."""
    },
    {
        "id": "doc_010",
        "topic": "Contrastive Learning and Self-Supervised Representations",
        "text": """Contrastive learning is a self-supervised representation learning technique that trains a model to
        distinguish similar (positive) pairs from dissimilar (negative) pairs without manual labels. SimCLR (Chen et al.,
        2020) creates two augmented views of each image and trains an encoder to maximise agreement between them using
        normalised temperature-scaled cross-entropy (NT-Xent) loss. MoCo (Momentum Contrast) maintains a dynamic
        queue of negative samples and uses a momentum encoder to produce consistent representations. CLIP (Contrastive
        Language-Image Pre-training by OpenAI) learns visual representations from natural language supervision by
        training on 400 million image-text pairs from the internet. It aligns image and text embeddings in a shared
        space using contrastive loss. CLIP achieves strong zero-shot transfer to downstream tasks. BYOL (Bootstrap
        Your Own Latent) and SimSiam learn without negative pairs using a bootstrap mechanism. Contrastive learning
        has been extended to text (SimCSE), graphs (GraphCL), and multi-modal settings. These methods learn rich,
        transferable features that are competitive with supervised pre-training on many benchmarks."""
    },
    {
        "id": "doc_011",
        "topic": "Neural Architecture Search (NAS)",
        "text": """Neural Architecture Search (NAS) automates the design of neural network architectures. Early NAS
        methods used reinforcement learning (Zoph & Le, 2017) or evolutionary algorithms to search over a large
        architecture space, but required thousands of GPU hours. One-shot NAS methods like DARTS (Differentiable
        Architecture Search) represent the search space as a single supernetwork and use gradient descent to
        optimise architectural parameters jointly with network weights, reducing search cost dramatically.
        EfficientNet used NAS combined with compound scaling to design a family of models that achieve
        state-of-the-art accuracy at various compute budgets. Hardware-aware NAS optimises for both accuracy
        and target hardware metrics like latency and memory footprint. ProxylessNAS and Single Path One-Shot
        methods further reduce memory requirements. NAS has been applied to CNNs, RNNs, Transformers, and
        mixed-precision quantisation. Limitations include the difficulty of defining appropriate search spaces,
        proxy tasks that do not always correlate with final performance, and the potential for found architectures
        to be hardware-specific."""
    },
    {
        "id": "doc_012",
        "topic": "Large Language Model Evaluation and Benchmarks",
        "text": """Evaluating Large Language Models (LLMs) requires comprehensive benchmarks covering diverse capabilities.
        GLUE and SuperGLUE benchmark NLU tasks like entailment, coreference resolution, and question answering.
        MMLU (Massive Multitask Language Understanding) tests knowledge across 57 subjects from elementary to
        professional level. HumanEval and MBPP assess code generation by measuring the percentage of problems
        that pass unit tests. BIG-Bench contains hundreds of tasks designed to probe capabilities beyond standard
        NLP benchmarks. TruthfulQA measures whether models generate truthful answers to questions humans often
        answer falsely. HellaSwag evaluates commonsense NLI and is easy for humans but challenging for models.
        MT-Bench and Chatbot Arena use GPT-4 or human judges to evaluate open-ended conversational quality.
        RAGAS evaluates RAG pipelines on faithfulness (does the answer follow from the context), answer relevancy
        (is the answer relevant to the question), and context precision (are retrieved documents relevant).
        Challenges in LLM evaluation include benchmark contamination (training data overlap), saturation of existing
        benchmarks, and the difficulty of measuring alignment with human values."""
    },
]


# ── 🧠 Build ChromaDB Knowledge Base ──────────────────────────────────────────

def build_knowledge_base():
    """
    Loads sentence-transformer embeddings and builds ChromaDB collection.
    MODIFY: Change the embedding model name below if you want a different one.
    Good alternatives: 'all-mpnet-base-v2' (better quality, slower)
                       'paraphrase-MiniLM-L6-v2' (lighter)
    """
    print("⏳ Loading embedding model (first run downloads ~90MB)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")  # MODIFY model name here

    client = chromadb.Client()
    try:
        client.delete_collection("research_paper_kb")
    except Exception:
        pass
    collection = client.create_collection("research_paper_kb")

    texts      = [d["text"]  for d in DOCUMENTS]
    ids        = [d["id"]    for d in DOCUMENTS]
    embeddings = embedder.encode(texts).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS],
    )
    print(f"✅ Knowledge base ready: {collection.count()} documents")
    return embedder, collection


# ── 📐 State Definition ───────────────────────────────────────────────────────

class CapstoneState(TypedDict):
    # Input
    question:       str
    # Memory
    messages:       List[dict]
    # Routing
    route:          str          # "retrieve" | "memory_only" | "tool"
    # RAG
    retrieved:      str
    sources:        List[str]
    # Tool
    tool_result:    str
    # Paper-specific (domain field)
    paper_title:    str          # extracted paper title if user mentions one
    # Answer
    answer:         str
    # Quality control
    faithfulness:   float
    eval_retries:   int


# ── 🔧 Constants ──────────────────────────────────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES       = 2


# ── 🏗️ Node Functions ─────────────────────────────────────────────────────────

def memory_node(state: CapstoneState) -> dict:
    """Adds current question to conversation history (sliding window of 6)."""
    msgs = state.get("messages", [])
    msgs = msgs + [{"role": "user", "content": state["question"]}]
    if len(msgs) > 6:
        msgs = msgs[-6:]
    return {"messages": msgs}


def router_node(state: CapstoneState, llm) -> dict:
    """
    Decides routing: retrieve / memory_only / tool.
    MODIFY: Update the prompt description below if you change the tool.
    """
    question = state["question"]
    messages = state.get("messages", [])
    recent   = "; ".join(
        f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]
    ) or "none"

    prompt = f"""You are a router for a Research Paper Q&A chatbot.

Available options:
- retrieve: search the knowledge base for information about AI/ML research papers and concepts
- memory_only: answer from conversation history (e.g. "what did you just say?", "can you explain that again?")
- tool: use web search when the user asks about very recent papers (after 2023), author profiles, paper citations, or links

Recent conversation: {recent}
Current question: {question}

Reply with ONLY one word: retrieve / memory_only / tool"""

    response = llm.invoke(prompt)
    decision = response.content.strip().lower()

    # Normalise LLM output
    if "memory" in decision:   decision = "memory_only"
    elif "tool" in decision:   decision = "tool"
    else:                      decision = "retrieve"

    # Extract paper title if mentioned (domain-specific enrichment)
    title_match = re.search(r'"([^"]+)"|\'([^\']+)\'', question)
    paper_title = (title_match.group(1) or title_match.group(2)) if title_match else ""

    return {"route": decision, "paper_title": paper_title}


def retrieval_node(state: CapstoneState, embedder, collection) -> dict:
    """Queries ChromaDB with semantic search. Returns top 3 chunks + sources."""
    q_emb   = embedder.encode([state["question"]]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=3)
    chunks  = results["documents"][0]
    topics  = [m["topic"] for m in results["metadatas"][0]]
    context = "\n\n---\n\n".join(
        f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks))
    )
    return {"retrieved": context, "sources": topics}


def skip_retrieval_node(state: CapstoneState) -> dict:
    """Used when routing to memory_only — no retrieval needed."""
    return {"retrieved": "", "sources": []}


def tool_node(state: CapstoneState) -> dict:
    """
    Web search tool using DuckDuckGo (free, no API key needed).
    Triggered when user asks about very recent papers or paper links.
    MODIFY: Replace with Hugging Face Inference API or any other tool.
    """
    question = state["question"]
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(
                f"research paper {question}",
                max_results=4
            ))
        if results:
            tool_result = "\n\n".join(
                f"Title: {r.get('title', 'N/A')}\n"
                f"Snippet: {r.get('body', 'N/A')[:300]}\n"
                f"URL: {r.get('href', 'N/A')}"
                for r in results
            )
        else:
            tool_result = "No web results found for this query."
    except ImportError:
        tool_result = (
            "Web search tool not available. "
            "Install with: pip install duckduckgo-search"
        )
    except Exception as e:
        tool_result = f"Web search error: {str(e)}"

    return {"tool_result": tool_result}


def answer_node(state: CapstoneState, llm) -> dict:
    """
    Generates the final answer using Gemini.
    Combines retrieved context + tool results + conversation history.
    MODIFY: Change the system prompt below to adjust agent behaviour.
    """
    question     = state["question"]
    retrieved    = state.get("retrieved", "")
    tool_result  = state.get("tool_result", "")
    messages     = state.get("messages", [])
    eval_retries = state.get("eval_retries", 0)
    paper_title  = state.get("paper_title", "")

    # Build context section
    context_parts = []
    if retrieved:
        context_parts.append(f"KNOWLEDGE BASE CONTEXT:\n{retrieved}")
    if tool_result:
        context_parts.append(f"WEB SEARCH RESULTS:\n{tool_result}")
    context = "\n\n".join(context_parts)

    # System prompt — MODIFY this for different behaviour
    if context:
        system_content = f"""You are a knowledgeable Research Paper Q&A assistant specialising in AI and Machine Learning research.
Your job is to answer questions about research papers, methodologies, authors, and concepts.

STRICT RULES:
1. Answer using ONLY the information in the context below.
2. If the answer is not fully in the context, say: "I don't have detailed information about that in my knowledge base. Try using the web search option."
3. Do NOT add information from your training data that is not in the context.
4. Cite the topic/paper name when you reference information.
5. If a paper title is mentioned ({paper_title if paper_title else 'none specified'}), focus your answer on that paper.

{context}"""
    else:
        system_content = """You are a helpful Research Paper Q&A assistant.
Answer based on the conversation history. If you cannot answer, say so clearly."""

    # Retry improvement instruction
    if eval_retries > 0:
        system_content += (
            "\n\nIMPORTANT: Your previous answer may have included information "
            "not in the context. Answer ONLY from what is explicitly stated above."
        )

    # Build LangChain message list
    lc_msgs = [SystemMessage(content=system_content)]
    for msg in messages[:-1]:
        lc_msgs.append(
            HumanMessage(content=msg["content"]) if msg["role"] == "user"
            else AIMessage(content=msg["content"])
        )
    lc_msgs.append(HumanMessage(content=question))

    response = llm.invoke(lc_msgs)
    return {"answer": response.content}


def eval_node(state: CapstoneState, llm) -> dict:
    """
    Self-reflection node: scores faithfulness of the answer.
    If score < FAITHFULNESS_THRESHOLD, triggers a retry in answer_node.
    """
    answer  = state.get("answer", "")
    context = state.get("retrieved", "")[:500]
    retries = state.get("eval_retries", 0)

    if not context:
        return {"faithfulness": 1.0, "eval_retries": retries + 1}

    prompt = f"""Rate faithfulness: does this answer use ONLY information from the context?
Reply with ONLY a decimal number between 0.0 and 1.0.
1.0 = fully grounded in context. 0.5 = some hallucination. 0.0 = mostly hallucinated.

Context: {context}
Answer: {answer[:300]}"""

    result = llm.invoke(prompt).content.strip()
    try:
        score = float(result.split()[0].replace(",", "."))
        score = max(0.0, min(1.0, score))
    except Exception:
        score = 0.5

    gate = "✅" if score >= FAITHFULNESS_THRESHOLD else "⚠️ retry"
    print(f"  [eval] Faithfulness: {score:.2f} {gate}")
    return {"faithfulness": score, "eval_retries": retries + 1}


def save_node(state: CapstoneState) -> dict:
    """Appends the assistant answer to the conversation history."""
    messages = state.get("messages", [])
    messages = messages + [{"role": "assistant", "content": state["answer"]}]
    return {"messages": messages}


# ── 🔀 Routing Functions ──────────────────────────────────────────────────────

def route_decision(state: CapstoneState) -> str:
    route = state.get("route", "retrieve")
    if route == "tool":        return "tool"
    if route == "memory_only": return "skip"
    return "retrieve"


def eval_decision(state: CapstoneState) -> str:
    score   = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)
    if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
        return "save"
    return "answer"  # retry


# ── 🏭 Agent Factory ──────────────────────────────────────────────────────────

def build_agent():
    """
    Builds and compiles the full LangGraph agent.
    Returns: (app, embedder, collection, llm)
    """
    llm               = get_llm()
    embedder, collection = build_knowledge_base()

    # Wrap nodes with their dependencies (closure pattern)
    def _router(state):   return router_node(state, llm)
    def _retrieval(state): return retrieval_node(state, embedder, collection)
    def _answer(state):   return answer_node(state, llm)
    def _eval(state):     return eval_node(state, llm)

    graph = StateGraph(CapstoneState)

    # Add all nodes
    graph.add_node("memory",   memory_node)
    graph.add_node("router",   _router)
    graph.add_node("retrieve", _retrieval)
    graph.add_node("skip",     skip_retrieval_node)
    graph.add_node("tool",     tool_node)
    graph.add_node("answer",   _answer)
    graph.add_node("eval",     _eval)
    graph.add_node("save",     save_node)

    # Entry point
    graph.set_entry_point("memory")
    graph.add_edge("memory", "router")

    # Router → branches
    graph.add_conditional_edges(
        "router", route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"}
    )

    # All paths → answer
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip",     "answer")
    graph.add_edge("tool",     "answer")

    # Eval gate
    graph.add_edge("answer", "eval")
    graph.add_conditional_edges(
        "eval", eval_decision,
        {"answer": "answer", "save": "save"}
    )
    graph.add_edge("save", END)

    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    print("✅ LangGraph agent compiled successfully!")
    print(f"   Nodes: memory → router → [retrieve|skip|tool] → answer → eval → save")
    print(f"   LLM: {LLM_MODEL}")
    print(f"   Documents: {len(DOCUMENTS)}")

    return app, embedder, collection, llm


# ── 🧪 Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing agent.py directly...")
    app, embedder, collection, llm = build_agent()
    result = app.invoke(
        {"question": "What is the Transformer architecture?"},
        config={"configurable": {"thread_id": "test-001"}}
    )
    print(f"\nAnswer: {result['answer'][:300]}...")
    print(f"Sources: {result.get('sources', [])}")
    print(f"Faithfulness: {result.get('faithfulness', 0):.2f}")
