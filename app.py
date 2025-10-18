# Paste this into app.py

import streamlit as st
from agent import create_financial_agent # Import our agent creator

# 1. Set up the page
st.set_page_config(page_title="Financial Analyst Agent", page_icon="📈")
st.title("📈 Financial Analyst Agent")
st.caption("Your AI-powered assistant for market insights and stock analysis")

# 2. Load API keys from Streamlit secrets
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    news_api_key = st.secrets["NEWS_API_KEY"]
except KeyError:
    st.error("ERROR: API keys not found. Please add them to .streamlit/secrets.toml")
    st.stop()

# 3. Cache the agent executor to prevent re-creating it on every re-run
@st.cache_resource
def get_agent_executor():
    return create_financial_agent(google_api_key, news_api_key)

try:
    agent_executor = get_agent_executor()
except Exception as e:
    st.error(f"Error creating agent: {e}")
    st.stop()

# 4. Initialize or get the chat history from session state
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hi! I'm your financial analyst. Ask me about a stock (e.g., 'How is NVIDIA (NVDA) doing today?')"
    }]

# 5. Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Get user input
if prompt := st.chat_input("What is your question?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display a spinner while the agent is "thinking"
    with st.spinner("Analyzing..."):
        try:
            # Call the agent
            response = agent_executor.invoke({"input": prompt})
            answer = response["output"]
        except Exception as e:
            answer = f"Sorry, an error occurred: {e}"
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    # Display assistant-response
    with st.chat_message("assistant"):
        st.markdown(answer)
