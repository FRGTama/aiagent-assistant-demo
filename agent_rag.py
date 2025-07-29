from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from dotenv import load_dotenv
import agent_tools
import models

load_dotenv()
system_prompt = """You are a smart assistant that can use tools when necessary, or answer questions directly when not.

    If a tool is needed, respond using the following format:
    Thought: I need to use a tool to get the answer.
    Action: <tool_name>
    Action Input: <input to the tool>
    
    If a tool is NOT needed, respond like this:
    Thought: I already know the answer.
    Final Answer: <your answer here>
    
    Only use one action at a time. Do not make up tools that are not available.
    """
agent = initialize_agent(
    tools=agent_tools.tool_list,
    llm=models.global_models.get_ocr_llm(),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_kwargs={"system_message": system_prompt},
    handle_parsing_errors=True,
    verbose=True
)

def call_agent(prompt):
    if prompt.lower() == "q":
        return "exited"
    result = agent.invoke(HumanMessage(prompt))
    return result
