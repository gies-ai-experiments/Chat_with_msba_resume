from fastapi import FastAPI
from dotenv import load_dotenv
import os
import gradio as gr

from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

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
            f"Give me 3 keywords from {name}'s resume related to {topic}"
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


with gr.Blocks() as demo:
    gr.Markdown("# Chat with MSBA Resume")
    with gr.Tab("Search"):
        gr.Markdown(
            "## Search students with the skills or experiences you are looking for"
        )
        search_input = gr.Textbox(label="Enter skills or experiences")
        search_button = gr.Button("Search")
        search_results = gr.Dataframe(
            headers=["Name", "Summary", "Resume"], interactive=False
        )

        def search_skills(query):
            students = get_student(query)
            data = [
                [student["name"], student["summary"], student["resume_link"]]
                for student in students
            ]
            return data

        search_button.click(search_skills, inputs=search_input, outputs=search_results)
    with gr.Tab("Chat"):
        chat_interface = gr.ChatInterface(fn=chatbot_response, title="MSBA Chatbot", chatbot=gr.Chatbot(render=False, height=500))

# with gr.Tab("Student Viewer"):
#     # student_dropdown = gr.Dropdown(
#     #     label="Select a Student", choices=[student.name for student in students]
#     # )  # List of student names
#     student_resume = gr.Document(label="Student Resume")
#     student_summary = gr.Textbox(label="Student Summary", interactive=False)

#     def view_student(student_name):

#         # resume_path, summary = get_student_info(student_name)
#         return "Hi"  # resume_path, summary

# student_dropdown.change(view_student, inputs=student_dropdown, outputs=[student_resume, student_summary])

# app = gr.mount_gradio_app(app, demo, path="/")
app = gr.mount_gradio_app(app, demo, path="/")