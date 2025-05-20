# Product Requirements Document (PRD) for LangGraph LiveKit Agents (Python)

## 1. Overview
The LangGraph LiveKit Agents project ("the agent") is a Python implementation that enables building voice-enabled AI agents using LangGraph and LiveKit. It offers a framework to assemble conversational pipelines, integrate LLMs, manage state, and stream audio interactions via LiveKit.

## 2. Objectives
- Provide reusable agent scaffolding for voice AI workloads
- Enable streaming, prompt-based interactions, and plug-in workflows
- Offer example pipelines demonstrating typical usage

## 3. Architecture & Components

### 3.1. Configuration & Environment
- `.env.example`  
  Template for environment variables (API keys, endpoints) required by the agent:
  - `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LIVEKIT_URL`, `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`
- `.gitignore`  
  Excludes Python artifacts, virtual environment folders, and sensitive files from version control.
- `Makefile`  
  Common task shortcuts:
  - `make start-agent` – run agent via Uvicorn
  - `make start-voice` – run voice-enabled server

### 3.2. Core Package Definition
- `pyproject.toml`  
  Build and dependency metadata (requires Python ≥3.12, LangGraph, spectral libraries)
- `uv.lock`  
  Locked dependency tree for deterministic installs

### 3.3. LangGraph Manifest
- `langgraph.json`  
  Defines graph mappings (agent → `example/agent.py`) and interactive dependencies

### 3.4. Example Usage
- `example/agent.py`  
  Sample agent script demonstrating:
  - Graph construction using LangGraphAdapter
  - Voice pipeline instantiation with Deepgram transcription & LiveKit streaming
  - Event handlers for “message”, “custom”, and “flush”
- `example/pipeline.py`  
  Demonstrates building a processing pipeline outside LiveKit context

### 3.5. Runtime Logic & Streaming
- `langgraph_livekit_agents/__init__.py`  
  Main adapter and runtime implementation:
  - `FlushSentinel`, `HumanMessage`, and helper classes
  - `LangGraphStream` orchestrates graph execution, calls LLMs, streams via HTTPX & LiveKit
  - Event loop handling for incoming messages, streaming chunks, and flushing
  - `_create_livekit_chunk` wraps chunks into LiveKit-compatible messages
- `langgraph_livekit_agents/types.py`  
  Defines typed stream writer and helper methods (`say`, `flush`)

## 4. Workflows & Integration
- Uses LangGraph core primitives (graphs, prompts, retrieval)
- Leverages HTTPX for asynchronous streaming to/from LLM backends
- Integrates Deepgram for real-time transcription and LiveKit for audio I/O

## 5. Testing
- Included tests validate stream-chunking and event ordering in the adapter.

## 6. Usage
```bash
uv pip install -e .
cp .env.example .env
# Fill in API and LiveKit credentials
make start-agent
```

## 7. Dependencies
- Python ≥3.12
- LangGraph (core graph SDK)
- livekit-agents SDK
- HTTPX, Uvicorn, deepgram-sdk

## 8. Future Enhancements
- Support additional voice backends (e.g., Whisper, Google Speech)
- Extend graph node ecosystem with custom analytics nodes
- Integrate authentication and authorization for multi-tenant usage