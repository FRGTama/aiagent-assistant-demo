from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from utils import format_convo
import os
load_dotenv()

model = OllamaLLM(model="llama3.2",base_url=os.getenv("LLM_ENDPOINT"))

template = ChatPromptTemplate([SystemMessage("""
You are a personal assistance, specializing in answering questions, 
and you can get better overtime by asking questions on topics you don't know, an evolving AI assistant,
if you don't know the answer to the user's question, 
clarify your incompetence and ask the user to give you more information so in the future if the user ask again you can answer it 
"""),
                               MessagesPlaceholder(variable_name="conversation", optional=True)])


def conversation_loop():
    convo = []
    while True:
        question = HumanMessage(content=input("\nq to quit: "))
        if question.content.lower() == "q":
            break
        convo.append(question)
        prompt = template.invoke({"conversation": convo})
        result = AIMessage(model.invoke(prompt))
        print(result.content)
        convo.append(result)
    return convo


print(format_convo(conversation_loop()))

