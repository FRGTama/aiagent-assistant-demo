from dotenv import load_dotenv
import gradio as gr
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import agent_rag_langgraph
import agent_tools
import agent_rag
load_dotenv()

#ocr_prompt = """perform ocr to extract chinese texts on the given pdf file consisting of scanned images of a chinese book which has vertical formatting,"""
#ocr_file_path = "../docs/testpdf.pdf"

# with gr.Blocks() as demo:
#     with gr.Row():
#         inp_img = gr.Image(type="filepath", height=200, width=500)
#         inp_text = gr.Textbox(placeholder="Raw text content")
#         inp_file = gr.File(type="filepath")
#         out_sum = gr.Markdown(container=True)
    # btn_trans = gr.Button("Translate")
    # btn_trans.click(fn=translate , inputs=inp_trans, outputs=out_sum)
    # btn_ocr = gr.Button("Scan image")
    # btn_ocr.click(fn=ocr_tool, inputs=inp_ocr, outputs=out_sum)
    # btn_sum = gr.Button("Summarize contents")
    # btn_sum.click(fn=summarize, inputs=inp_sum, outputs=out_sum)
def prompt_agent(img_path, prompt, file_path):
    sys_prompt = f"""You are an assistant specialize in answering questions, given are the image file path named img_path
    a pdf file path named file_path and a prompt in string format, answer the prompt based on info gained from 
    the image or the pdf file.
    img_path: {img_path},
    file_path: {file_path},
    prompt: {prompt}
"""

    return agent_rag.call_agent(sys_prompt)["output"]

# agent_demo = gr.Interface(
#     fn=prompt_agent,
#     inputs= [gr.Image(type="filepath", height=200, width=500),
#         gr.Textbox(placeholder="Raw text content"),
#         gr.File(type="filepath")],
#     outputs= gr.Markdown(container=True)
# )
#
# agent_demo.launch(share=True)
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):

        bot_message = agent_rag_langgraph.stream_graph_updates(message)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
            # time.sleep(2)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()