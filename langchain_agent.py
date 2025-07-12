from langchain_core.messages import HumanMessage
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import TavilySearchResults

from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

load_dotenv()
import os

search_tool = TavilySearchResults(search_depth="basic")

agent_tools = [search_tool]

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
llm = OllamaLLM(model="mannix/jan-nano",base_url=os.getenv("LOCAL_ENDPOINT"))

agent = initialize_agent(
    tools=agent_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run the agent
agent.invoke([HumanMessage("Should i bring an umbrella today, i live in Thủ Đức")])