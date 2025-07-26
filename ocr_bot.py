from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import base64
from dotenv import load_dotenv
import pymupdf


load_dotenv()

ocr_bot = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
def ocr(image_file_path):
    with open(image_file_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    message_local = HumanMessage(
        content=[
            {"type": "text", "text": "Extract all text from the given image, return the raw text."},
            {"type": "image_url", "image_url": f"data:image/png;base64,{encoded_image}"},
        ]
    )
    result_local = ocr_bot.invoke([message_local])
    return result_local.content

## extract images from pdf to .output/ocr/. and returns number of page
def extract_text_from_pdf(path):
    doc = pymupdf.open(path)  # open document
    content = []
    for page in doc:  # iterate through the pages
        pix = page.get_pixmap()  # render page to an image
        pix.save("./output/ocr/page-%i.png" % page.number)  # store image as a PNG
        content.append([ocr("./output/ocr/page-%i.png" % page.number)])
    return content


