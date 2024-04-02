from fastapi import FastAPI
from dotenv import load_dotenv
import os
import gradio as gr
from pathlib import Path
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from typing import Any, List

import fitz
from PIL import Image

load_dotenv()
app = FastAPI()

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_EMBEDDING = os.getenv("AZURE_EMBEDDING")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_EMBEDDING,
    openai_api_version="2024-03-01-preview",
)
llm = AzureChatOpenAI(
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    openai_api_version="2024-03-01-preview",
    temperature=0.1,
)

vector_store = PineconeVectorStore(index_name='msba', embedding=embeddings)
retriever = vector_store.as_retriever()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            As an AI assistant, your primary role is to help me answer questions about MSBA students and their skills and experiences. 
            Answer the question using ONLY the following context. If you don't know the answer, 
            just say you don't know. DO NOT make up an answer.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


def format_docs(docs):
    return "\n\n".join(documents.page_content for documents in docs)


chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)
summary = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            As an AI assistant, you will give 3 keywords from student's resume. Do not write full sentences. Just need words.
            Answer the question using ONLY the following context. If you don't know the answer, 
            just say you don't know. DO NOT make up an answer.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)
chain2 = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | summary
    | llm
)


def get_student(topic: str):
    docs_scores = vector_store.similarity_search_with_score(topic, 7)
    recommendations = []
    for doc_score in docs_scores:
        doc, score = doc_score  # Assuming doc_score is a tuple like (doc, score)
        # Now, let's assume doc is an object with a 'metadata' attribute which is a dictionary
        name = doc.metadata["source"].split(".pdf")[0]  # Example adjustment
        response = chain2.invoke(
            f"Give me 3 keywords from {name}'s resume. Try relating to {topic} but if there isn't any just list what stands out"
        )
        summary = response.content
        resume_link = f"View Resume of {name}"
        recommendations.append(
            {"name": name, "summary": summary, "resume_link": resume_link}
        )
    return recommendations


def chatbot_response(messages, hisory):

    response = chain.invoke(messages)

    return response.content

PDF_FOLDER = 'msba'

def list_pdf_files(search_query=""):
    """
    List PDF files in the 'msba' directory filtered by the search query.
    """
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    if search_query:
        pdf_files = [f for f in pdf_files if search_query.lower() in f.lower()]
    # Return as a list of lists for DataFrame compatibility
    return [[f] for f in pdf_files]

def render_pdf_file(file_name, page_number=0):
    """
    Render a specific page of a PDF file as an image.
    """
    file_path = os.path.join(PDF_FOLDER, file_name)
    doc = fitz.open(file_path)
    if page_number < len(doc):
        page = doc.load_page(page_number) 
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    return None
def list_pdf_filenames(search_query=""):
    """
    List PDF filenames for the dropdown selection.
    """
    return [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf") and (search_query.lower() in f.lower() if search_query else True)]

def display_pdf_from_name(pdf_name):
    """
    Display the PDF based on the selected name from the Dropdown.
    """
    if pdf_name:
        return render_pdf_file(pdf_name)
    return None


with gr.Blocks() as demo:
    gr.Markdown("# Chat with MSBA Resume")
    
    with gr.Tab("Chat"):
        chat_interface = gr.ChatInterface(fn=chatbot_response, title="MSBA Chatbot", chatbot=gr.Chatbot(render=False, height=500))
        
    with gr.Tab("Search"):
        gr.Markdown(
            "## Search students with the skills or experiences you are looking for"
        )
        search_input = gr.Textbox(label="Enter skills or experiences")
        search_button = gr.Button("Search")
        search_results = gr.Dataframe(
            headers=["Name", "Summary"], interactive=False
        )

        def search_skills(query):
            students = get_student(query)
            data = [
                [student["name"], student["summary"]]
                for student in students
            ]
            return data

        search_button.click(search_skills, inputs=search_input, outputs=search_results)
        
    with gr.Tab("Resumes"):
        with gr.Row():
            with gr.Column():
                show_img = gr.Image(label='PDF Preview')
            with gr.Column():
                pdf_dropdown = gr.Dropdown(label="Select a PDF", choices=list_pdf_filenames())
                show_button = gr.Button("Show PDF")
                
                show_button.click(fn=display_pdf_from_name, inputs=[pdf_dropdown], outputs=[show_img])




app = gr.mount_gradio_app(app, demo, path="/")