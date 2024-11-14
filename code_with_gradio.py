import gradio as gr
import os
import moviepy.editor as mp
import speech_recognition as sr
from fpdf import FPDF
from transformers import pipeline
import fitz  # PyMuPDF
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
import torch  # Importing torch for device configuration

# Set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to handle the dropdown selection
def process_input(choice, file_path):
    if choice == "Convert Video to Summarized PDF":
        if os.path.exists(file_path):
            # Video to Audio
            audio_path = video_to_audio(file_path)
            # Audio to Text
            text = audio_to_text(audio_path)
            # Summarize Text
            summary = summarize_text(text)
            # Save summary to PDF
            save_text_to_pdf(summary, "video_summary.pdf")
            return summary, "video_summary.pdf"
        else:
            return "Invalid video file path.", None

    elif choice == "Convert Audio to Summarized PDF":
        if os.path.exists(file_path):
            # Audio to Text
            text = audio_to_text(file_path)
            # Summarize Text
            summary = summarize_text(text)
            # Save summary to PDF
            save_text_to_pdf(summary, "audio_summary.pdf")
            return summary, "audio_summary.pdf"
        else:
            return "Invalid audio file path.", None

    elif choice == "Summarize existing PDF":
        if os.path.exists(file_path):
            # Extract text from PDF
            text = extract_text_from_pdf(file_path)
            # Summarize Text
            summary = summarize_text(text)
            # Save summary to PDF
            save_text_to_pdf(summary, "pdf_summary.pdf")
            return summary, "pdf_summary.pdf"
        else:
            return "Invalid PDF file path.", None

# Helper functions from your previous code

def video_to_audio(video_path, audio_output="converted_audio.wav"):
    clip = mp.VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_output)
    return audio_output

def audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Speech Recognition could not understand the audio."
    except sr.RequestError:
        return "Could not request results from Speech Recognition service."

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def summarize_text(text, chunk_size=1024):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
    summaries = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return " ".join(summaries)

def save_text_to_pdf(text, pdf_output):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    text = text.encode('latin-1', 'replace').decode('latin-1')
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf.output(pdf_output)

# Gradio UI
def interactive_qa_system(pdf_file):
    MODEL = "llama3"
    
    model = Ollama(model=MODEL)
    embeddings = OllamaEmbeddings(model=MODEL)
    parser = StrOutputParser()

    template = """
    Answer the question based on the context below. If you can't
    answer the question, answer with "I don't know".

    Context: {context}

    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)

    loader = PyPDFLoader(pdf_file)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    text_documents = text_splitter.split_documents(pages)[:5]

    vectorstore = DocArrayInMemorySearch.from_documents(text_documents, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    def qa_answer(question):
        retrieved_context = retriever.invoke(question)
        formatted_prompt = prompt.format(context=retrieved_context, question=question)
        response_from_model = model.invoke(formatted_prompt)
        parsed_response = parser.parse(response_from_model)
        return parsed_response

    return qa_answer

# Define Gradio interface
with gr.Blocks() as demo:

    gr.Markdown("<h1 style='text-align: center;'>Automated Multimedia Summarization and Translation System</h1>")

    choice = gr.Dropdown(["Convert Video to Summarized PDF", "Convert Audio to Summarized PDF", "Summarize existing PDF"], label="Choose an option")
    file_path = gr.Textbox(label="Enter the path to the file")
    
    summary_output = gr.Textbox(label="Summary", interactive=False)
    summary_file = gr.File(label="Download PDF", visible=True)
    
    summary_button = gr.Button("Generate Summary")
    
    summary_button.click(fn=process_input, inputs=[choice, file_path], outputs=[summary_output, summary_file])
    
    question = gr.Textbox(label="Ask a question about the summary")
    answer_output = gr.Textbox(label="Answer", interactive=False)
    question_button = gr.Button("Ask Question")
    
    qa_system = interactive_qa_system("pdf_summary.pdf")  # Assuming the PDF is already summarized
    question_button.click(fn=qa_system, inputs=question, outputs=answer_output)

demo.launch()
