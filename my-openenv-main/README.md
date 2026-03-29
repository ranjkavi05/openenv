---
title: LifeOS ✨ 🌌
emoji: 🌌
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.30+-red?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/OpenEnv-Compatible-green?style=for-the-badge" alt="OpenEnv">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p>

<h1 align="center">🌌 LifeOS ✨</h1>

<p align="center">
  <strong>A reinforcement-learning environment that simulates real-world human life decisions.</strong><br>
  Balance health, career, relationships, finances, and stress — how well can you live?
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#example-output">Example Output</a> •
  <a href="#deployment">Deployment</a>
</p>

---

## 🧩 Problem Statement

Real-life decisions are interconnected — working overtime earns money but costs health and relationships. Traditional RL environments rarely capture this **multidimensional trade-off**. The LifeOS ✨ acts as a **Personal Resource Allocation Optimizer** where an AI agent must learn to balance competing priorities across 3 distinct tasks (Wealth, Career, and Balance) to achieve a fulfilling life, strictly adhering to the OpenEnv specification.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **OpenEnv API Compliant** | FastAPI server with `/reset` & `/step` endpoints |
| 🧠 **6 Actions** | Work, Exercise, Invest, Learn, Socialize, Rest |
| 📊 **7 State Variables** | Age, Health, Money, Stress, Career, Relationships, Happiness |
| 🏆 **3 Distinct Tasks** | Wealth Builder (Easy), Career Climber (Medium), Perfect Balance (Hard) |
| 🎭 **5 Personality Modes** | Risk Taker, Conservative, Lazy, Ambitious, Balanced |
| 🌪️ **8 Random Events** | Promotions, layoffs, emergencies, market crashes, and more |
| ⏱️ **Time Progression** | Aging, health decay, delayed investment returns |
| 🎚️ **3 Difficulty Levels** | Easy, Medium, Hard with increasing complexity |
| 🏆 **Agent Grading** | 0.0–1.0 normalized life-quality score |
| 🤖 **Baseline Agent** | Rule-based AI with decision explanations |
| 🧠 **LLM-Powered Agent** | OpenAI-compatible LLM agent via `inference.py` |
| 🎨 **Stunning Dashboard** | Glassmorphism UI with dark/light mode, Plotly charts |
| ✨ **Neon OS Aesthetic** | Custom cyberpunk gradient typography and responsive sidebar |
| 🚀 **One-Command Deploy** | Docker + Hugging Face Spaces ready |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    inference.py (Entry Point)                │
│         LLM Agent ← OpenAI Client (API_BASE_URL,            │
│                      MODEL_NAME, HF_TOKEN)                  │
├─────────────────────────────────────────────────────────────┤
│                api.py (FastAPI) & app.py (UI)                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ /reset   │ │ /step    │ │ /state   │ │   Dashboard   │  │
│  │ Endpoint │ │ Endpoint │ │ Endpoint │ │     UI        │  │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      env.py (Core)                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │  State   │ │  Reward  │ │   Time   │ │  Difficulty   │  │
│  │  Engine  │ │  System  │ │  System  │ │   Config      │  │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────┘  │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│models.py │events.py │person-   │grader.py │   agent.py      │
│  Enums   │ Dynamic  │alities  │ Scoring  │  Rule-Based     │
│  Types   │ Events   │  .py    │ Grades   │  Baseline       │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│                     utils.py (Helpers)                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ai-digital-life-simulator.git
cd ai-digital-life-simulator

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Before running the inference script, set the following environment variables:

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | The API endpoint for the LLM |
| `MODEL_NAME` | The model identifier to use for inference |
| `HF_TOKEN` | Your Hugging Face / API key |

```bash
# Example: set environment variables
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
export HF_TOKEN="hf_your_token_here"
```

> **Note:** If these variables are not set, the inference script will automatically fall back to a deterministic rule-based agent.

### Run the Inference Script (Required for Submission)

```bash
python inference.py
```

This runs all 3 tasks (`wealth_builder`, `career_climber`, `perfect_balance`) using the LLM-powered agent via the OpenAI Client, producing grader scores in the 0.0–1.0 range.

### Run the Server and Dashboard

The latest version includes a FastAPI server for strict OpenEnv compliance.

```bash
# On Linux / macOS (runs both API and Streamlit)
./start.sh

# On Windows (use two separate terminals)
# Terminal 1: Launch the API
uvicorn api:app --port 8000
# Terminal 2: Launch the dashboard
streamlit run app.py
```

- **Dashboard:** [http://localhost:8501](http://localhost:8501)
- **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Run the Baseline Agent (Terminal)

```bash
python agent.py
```

---

## 🎮 How It Works

### Actions & Effects

Each action affects multiple life dimensions, creating realistic trade-offs:

| Action | 💰 Money | ❤️ Health | 😰 Stress | 💼 Career | 👥 Social | 😊 Happy |
|--------|---------|---------|---------|---------|---------|---------|
| Work Overtime | +120 | -3 | +12 | +4 | -2 | -4 |
| Exercise | -10 | +8 | -8 | — | +1 | +5 |
| Invest Money | ±var | — | +4 | +1 | -1 | — |
| Learn Skill | -30 | -1 | +5 | +7 | -1 | +3 |
| Socialize | -40 | +1 | -5 | +1 | +10 | +8 |
| Rest | -5 | +5 | -12 | -1 | +2 | +4 |

### Reward System

The reward function encourages **balanced living**, not extreme min-maxing:

1. **Weighted Score** — Health (25%), Career (20%), Relationships (20%), Money (15%), Low Stress (20%)
2. **Imbalance Penalty** — Standard deviation penalty across dimensions
3. **Consistency Bonus** — Sustained balance over 5 consecutive steps
4. **Burnout Penalty** — Escalating penalty for chronic high stress

### Random Events

Probabilistic life events that affect multiple variables:

| Event | Probability | Key Effects |
|-------|------------|-------------|
| 🎉 Job Promotion | 6% | Career +15, Money +300 |
| 😰 Job Loss | 4% | Career -20, Stress +20 |
| 🏥 Medical Emergency | 5% | Health -25, Money -500 |
| 📉 Market Crash | 4% | Money -400, Stress +15 |
| 👨‍👩‍👧 Family Issue | 6% | Relationships -15, Stress +12 |
| 🎰 Lottery Win | 2% | Money +800, Happiness +15 |
| 🌪️ Natural Disaster | 3% | Health -10, Money -300 |
| 💰 Inheritance | 2% | Money +1000, Relationships +5 |

### Termination Conditions

- ❌ Health reaches 0 (death)
- ❌ Stress reaches 100 (total burnout)
- ✅ Maximum steps reached (simulation ends)

---

## 📊 Example Output

### Inference Script (`python inference.py`)

```
============================================================
  AI DIGITAL LIFE SIMULATOR - Inference Script
============================================================
  API_BASE_URL : https://api-inference.huggingface.co/v1
  MODEL_NAME   : meta-llama/Meta-Llama-3-8B-Instruct
  HF_TOKEN     : ********...abcd
============================================================

  Running task: wealth_builder
  --------------------------------------------------
    Step  25 | Action: work_overtime   | Reward: +0.0120 | Cumulative: -0.0000
    Step 100 | Action: invest_money    | Reward: -0.0150 | Cumulative: +0.0897
  -- Result: Grade = 0.0379  |  [!] Critical - Life in Crisis

  Running task: career_climber
  --------------------------------------------------
    Step 100 | Action: learn_skill     | Reward: +0.2700 | Cumulative: +3.1700
  -- Result: Grade = 1.0000  |  [*] Excellent - Balanced & Thriving

  Running task: perfect_balance
  --------------------------------------------------
    Step 100 | Action: work_overtime   | Reward: +0.4753 | Cumulative: +41.1787
  -- Result: Grade = 0.5182  |  [~] Average - Room for Improvement

============================================================
  FINAL RESULTS SUMMARY
============================================================
  [PASS] wealth_builder       | Grade: 0.0379 | Steps: 100
  [PASS] career_climber       | Grade: 1.0000 | Steps: 100
  [PASS] perfect_balance      | Grade: 0.5182 | Steps: 100

  Total time: 0.4s
  All scores in 0.0-1.0 range: YES
============================================================
```

---

## 🐳 Deployment

### Docker

```bash
# From the project root
docker build -t life-sim .
docker run -p 8000:8000 -p 8501:8501 life-sim
```

### Hugging Face Spaces

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. **Important:** Select **Docker** as the SDK (Do NOT select Streamlit). 
   - *Why?* If you select Streamlit, Hugging Face will only run the UI dashboard and will ignore the FastAPI server, causing your OpenEnv automated grading API to fail. By selecting Docker, Hugging Face will run the provided `Dockerfile` which launches both the API and the UI concurrently via `start.sh`.
3. Upload all project files to the repository.
4. The Space will automatically build and deploy the container.

---

## 📂 Project Structure

```
OpenEnv_Soln/
├── inference.py        # ** Main inference script (required for submission) **
├── api.py              # FastAPI server (OpenEnv REST compliance)
├── start.sh            # Launch script for API & UI
├── app.py              # Streamlit UI dashboard
├── env.py              # Core environment (reset/step/state)
├── models.py           # Data models, enums, dataclasses
├── utils.py            # Utility functions
├── events.py           # Dynamic random events system
├── personalities.py    # Personality modifier profiles
├── grader.py           # Agent grading (0.0-1.0)
├── agent.py            # Baseline rule-based AI agent
├── openenv.yaml        # OpenEnv specification
├── style.css           # Glassmorphism CSS theme
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker deployment
├── README.md           # This file
└── .streamlit/
    └── config.toml     # Streamlit server config
```

---

## 🛠️ Tech Stack

- **Python 3.11** — Core language
- **OpenAI Client** — LLM-based agent decisions (via `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`)
- **FastAPI / Uvicorn** — Backend REST API providing `/reset` and `/step`
- **Streamlit** — Interactive web dashboard
- **Plotly** — Beautiful interactive charts
- **CSS3** — Glassmorphism, animations, theming
- **Docker** — Containerized deployment

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ for the Hackathon<br>
  <strong>LifeOS ✨</strong> — Navigate life's decisions. Balance your destiny.
</p>
