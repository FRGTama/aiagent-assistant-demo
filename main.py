from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv

from memory import Memory

from mem_sentinel import reflection_template, episodic_recall, add_episodic_mem, episodic_sys_prompt
from utils import format_convo
import os

load_dotenv()

default_prompt = SystemMessage("""
You are a personal AI that will help the user in succeeding their academic career, by evolving over interactions,
try your best to help them understand the problem given, 
respond in short converse forms like you are talking and only give detailed explanation when asked to do so,
if you don't know a question, say you don't know paired with a question that helps you understand better, this is the questions:
""")

llmmodel = OllamaLLM(model="mannix/jan-nano",base_url=os.getenv("LOCAL_ENDPOINT"))

mem = Memory()
def conversation_loop():
    convo = [BaseMessage(''), default_prompt]
    while True:
        question = HumanMessage(content=input("\nq to quit: "))
        if question.content.lower() == "q":
            break
        if question.content.lower() == "q_quiet":
            break
        convo.append(question)
        result = AIMessage(llmmodel.invoke(convo))
        print(result.content)
        convo.append(result)
    return convo
res = conversation_loop()
add_episodic_mem(mem,res)
# print('\n\n\n\n')
print(format_convo(res))

# add_episodic_mem(mem,res)
# print(episodic_recall(mem0,"what do you know about me?")["documents"])

