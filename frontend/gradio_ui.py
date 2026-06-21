from dotenv import load_dotenv
import gradio as gr
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import agent_tools
import agent_rag
load_dotenv()

#ocr_prompt = """perform ocr to extract chinese texts on the given pdf file consisting of scanned images of a chinese book which has vertical formatting,"""
#ocr_file_path = "../docs/testpdf.pdf"

def prompt_agent(prompt, history, file, image):
    img_path = image if image else "None"
    file_path = file if file else "None"
    sys_prompt = f"""You are an assistant specialize in answering questions,
    given are the image file path named img_path
    a pdf file path named file_path and a prompt in string format, 
    if any of the argument are listed as None, then skip the processing of such path.
    answer the prompt based on info gained from 
    the image or the pdf file if there are paths, else answer based on your own knowledge and tools available,
    If you lack info on what the user wants or need more info to use a tool, ask the user for such information.
    img_path: {img_path},
    file_path: {file_path},
    chat history: {[msg['content'] for msg in history if msg['role'] == 'user']}
    prompt: {prompt}
"""
    result = agent_rag.call_agent(sys_prompt)["output"]

    return result

chat_ui = gr.ChatInterface(
    fn=prompt_agent,
    chatbot=gr.Chatbot(type="messages", height=400 ),
    additional_inputs=[
        gr.Image(type="filepath", label="Image"),
        gr.File(type="filepath", label="PDF")
    ],
    title="NotebookVL",
    description="Ask questions using an image and/or PDF file as context.",
)

chat_ui.launch()
