from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize model separately
def create_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0
    )
config = {"configurable": {"thread_id": "1"}}

def chatbot(state: State):
    llm = create_llm()  # Create model instance inside function
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)
# graph = graph_builder.compile()

def gradio_chatbot(message, history):
    # Convert to LangGraph format
    messages = []
    for human, ai in history:
        messages.append(HumanMessage(content=human))
        if ai:
            messages.append(response)  # Add AI message

    messages.append(HumanMessage(content=message))

    # Get response from your LangGraph
    result = graph.invoke({"messages": messages})

    # Extract just the text content for Gradio
    ai_response = result["messages"][-1].content

    return ai_response  # Return just the text, not the whole message object


