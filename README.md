# Automated Multimedia Summarization and Translation System

This project is an automated multimedia summarization and translation system. It allows users to convert video and audio files into summarized PDFs, extract and summarize content from existing PDFs, and perform interactive Q&A on the summaries. The project utilizes deep learning models and NLP tools, and it offers a user-friendly interface powered by **Gradio** for easy interaction.

![summary](https://github.com/user-attachments/assets/5ce82503-ef4a-45b4-8eed-26731fca62ac)
![summary2](https://github.com/user-attachments/assets/9b265090-608e-4085-a0c8-2d0c280b6f62)


## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Features

- **Video to Summarized PDF**: Converts video files to audio, transcribes the audio to text, and summarizes the text, saving it in a PDF.
- **Audio to Summarized PDF**: Transcribes audio files to text, summarizes the text, and saves it in a PDF.
- **PDF Summarization**: Extracts and summarizes text from an existing PDF, saving the summary in a new PDF.
- **Interactive Q&A**: Allows users to ask questions about the summarized PDF content and get relevant answers.

## Technologies Used

- **Gradio**: Provides an interactive user interface.
- **MoviePy**: For converting video to audio.
- **SpeechRecognition**: For transcribing audio files to text.
- **FPDF**: For creating PDF files.
- **Hugging Face Transformers**: For text summarization using BART.
- **LangChain**: Used for document retrieval and Q&A functionality.
- **Torch**: For GPU support in summarization.

## Installation

### Prerequisites

- Python 3.8 or higher
- GPU-enabled environment (optional but recommended for faster processing)

### Steps to Install

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/automated-summarization-translation.git
   cd automated-summarization-translation

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Start the Gradio interface**:
   ```bash
   python app.py
   ```

2. **Select an option from the dropdown**:

- Convert Video to Summarized PDF
- Convert Audio to Summarized PDF
- Summarize existing PDF

3. **Provide the file path** in the textbox and click "Generate Summary".

4. **Ask a Question:** After generating a summary, enter a question related to the content in the summary textbox and get an answer.

## Project Structure

```plaintext
Automated_Multimedia_Summarization/
│
├── app.py                         # Main application file for Gradio interface
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
│
├── helpers/                       # Helper scripts for each processing function
│   ├── video_to_audio.py          # Converts video to audio
│   ├── audio_to_text.py           # Transcribes audio to text
│   ├── extract_text_from_pdf.py   # Extracts text from PDF
│   ├── summarize_text.py          # Summarizes text using transformers
│   ├── save_text_to_pdf.py        # Saves summarized text to PDF
│
└── output/                        # Output folder for saved PDFs
```

- **app.py:** Main script to launch the Gradio interface.
- **helpers/:** Contains helper scripts for multimedia conversion, transcription, summarization, and PDF generation.
- **output/:** Folder where generated summary PDFs are saved.
