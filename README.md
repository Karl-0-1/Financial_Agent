# 📈 AI Financial Analyst Agent

This project is a conversational AI agent that acts as a junior financial analyst. It uses the LangChain framework with Google's Gemini models to answer complex financial questions.

The agent can autonomously use a set of tools to:
* Fetch live stock data (`yfinance`)
* Search for recent financial news (`NewsAPI`)
* Perform sentiment analysis on news headlines (using `Gemini`)

This project is built with Streamlit for an interactive web interface.

## 🤖 Demo Screenshot

`![App Screenshot](demo.png)`

## ✨ Features

* **Tool-Using Agent:** Built with LangChain's `create_tool_calling_agent` for robust reasoning.
* **Real-time Data:** Integrates `yfinance` for up-to-the-minute stock prices and `NewsAPI` for the latest news.
* **AI-Powered Sentiment:** Uses a separate Gemini model to analyze the sentiment of news articles.
* **Chat Interface:** A simple and clean chat UI built with Streamlit, maintaining conversation history.
* **Efficient:** Uses the fast `gemini-flash-latest` model for all AI tasks to ensure quick responses.

## 🛠️ Tech Stack

* **Language:** Python
* **GenAI Framework:** LangChain
* **LLMs:** `gemini-flash-latest` (for both agent reasoning and sentiment analysis)
* **Web Framework:** Streamlit
* **Data Sources:** yfinance, NewsAPI

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone [https://github.com/Karl-0-1/Financial_Agent.git](https://github.com/Karl-0-1/Financial_Agent.git)
cd Financial_Agent
