# Paste this into agent.py

import json
import yfinance as yf
from newsapi import NewsApiClient
from langchain.tools import tool, Tool  # <-- CHANGE 1: Import 'Tool'
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Tool 1: Stock Info (This one is simple, so @tool is fine) ---
@tool
def get_stock_info(ticker: str):
    """
    Fetches the current stock price, day high, day low, and trading volume
    for a given stock ticker using yfinance.
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        
        if data.empty:
            return f"Error: No data found for ticker {ticker}. It might be delisted or incorrect."
        
        latest = data.iloc[-1]
        info = {
            "ticker": ticker,
            "current_price": latest['Close'],
            "day_high": latest['High'],
            "day_low": latest['Low'],
            "volume": latest['Volume']
        }
        return f"Successfully fetched stock data for {ticker}: {json.dumps(info)}"
    
    except Exception as e:
        return f"Error fetching stock data for {ticker}: {e}"

# --- Tool 2: Financial News (Needs key, so we remove @tool) ---
# <-- CHANGE 2: Removed @tool decorator
def get_financial_news(query: str, news_api_key: str):
    """
    Fetches the top 5 recent financial news headlines for a given company or query.
    Requires a NewsAPI key.
    """
    try:
        newsapi = NewsApiClient(api_key=news_api_key)
        top_headlines = newsapi.get_everything(
            q=query,
            language='en',
            page_size=5,
            sort_by='relevancy'
        )
        
        if top_headlines['totalResults'] == 0:
            return f"No news articles found for '{query}'."
        
        articles = [article['title'] for article in top_headlines['articles']]
        return f"Successfully fetched news for {query}: {json.dumps(articles)}"
    
    except Exception as e:
        return f"Error fetching news for {query}: {e}"

# --- Tool 3: Sentiment Analysis (Needs key, so we remove @tool) ---
# <-- CHANGE 2: Removed @tool decorator
def analyze_sentiment(headlines: str, google_api_key: str):
    """
    Analyzes the sentiment of a list of news headlines (provided as a single string)
    and returns a sentiment classification (Positive, Negative, or Neutral) 
    with a justification. Requires a Google API key for the LLM.
    """
    try:
        sentiment_llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest", 
            google_api_key=google_api_key,
            temperature=0
        )
        
        sentiment_prompt = ChatPromptTemplate.from_template(
            """Analyze the sentiment of the following news headlines.
            Classify the overall sentiment as 'Positive', 'Negative', or 'Neutral'.
            Provide a one-sentence justification for your classification.

            Headlines:
            {headlines}

            Your analysis (Sentiment: [Positive/Negative/Neutral], Justification: [Your one-sentence justification]):
            """
        )
        
        sentiment_chain = sentiment_prompt | sentiment_llm | StrOutputParser()
        return sentiment_chain.invoke({"headlines": headlines})
    
    except Exception as e:
        return f"Error analyzing sentiment: {e}"

# --- Main Agent Creation Function ---
def create_financial_agent(google_api_key: str, news_api_key: str):
    """
    Creates and returns the financial analyst agent executor.
    """
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        google_api_key=google_api_key,
        temperature=0.2
    )

    # <-- CHANGE 3: Manually wrap lambda functions in Tool() objects -->
    tools = [
        get_stock_info, # This one is already a tool from the @tool decorator
        
        Tool(
            name="get_financial_news",
            func=lambda query: get_financial_news(query, news_api_key=news_api_key),
            description="Fetches the top 5 recent financial news headlines for a given company or query. Requires a NewsAPI key."
        ),
        
        Tool(
            name="analyze_sentiment",
            func=lambda headlines: analyze_sentiment(headlines, google_api_key=google_api_key),
            description="Analyzes the sentiment of a list of news headlines (provided as a single string) and returns a sentiment classification (Positive, Negative, or Neutral) with a justification. Requires a Google API key for the LLM."
        )
    ]

    # The rest of the function is the same
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a helpful and cautious junior financial analyst.
            Your goal is to provide a comprehensive answer to the user's question.
            You must use your tools to gather all necessary information.

            Here is your required workflow:
            1.  First, use the `get_financial_news` tool to find recent news for the company.
            2.  Second, take the news headlines from step 1 and use the `analyze_sentiment` tool.
            3.  Third, use the `get_stock_info` tool to get the latest stock price data.
            4.  Finally, synthesize all this information (news, sentiment, and stock data) 
                into a single, comprehensive answer.

            **IMPORTANT RULES:**
            - You must explicitly state that this is not financial advice.
            - Always add this disclaimer at the very end of your response:
              "Note: This is not financial advice. Please conduct your own research before making any investment decisions."
            """),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    
    agent = create_tool_calling_agent(llm, tools, agent_prompt)

    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True
    )
    
    return agent_executor
