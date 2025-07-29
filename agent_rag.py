from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from dotenv import load_dotenv
import agent_tools
import models

load_dotenv()

agent = initialize_agent(
    tools=agent_tools.tool_list,
    llm=models.global_models.ocr_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)

def call_agent(prompt):
    if prompt.lower() == "q":
        return "exited"
    result = agent.invoke(HumanMessage(prompt))
    return result
