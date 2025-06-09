import gradio as gr
import numpy as np
import faiss
import nltk
import os
from nltk.tokenize import sent_tokenize
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Download required nltk data
nltk.download("punkt")

# Load lightweight models
asr_model = WhisperModel("base", compute_type="int8")  # Fast and light
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Global variables for storing chunks and their embeddings
chunks = []
chunk_index = None

# Transcription and chunking
def transcribe_audio(audio_file):
    global chunks, chunk_index

    segments, _ = asr_model.transcribe(audio_file)
    transcript = " ".join([seg.text for seg in segments])

    sentences = sent_tokenize(transcript)
    chunk_size = 5
    chunks = [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    chunk_index = faiss.IndexFlatL2(embeddings.shape[1])
    chunk_index.add(embeddings)

    return transcript

# Question answering
def answer_question(question):
    if not chunks or chunk_index is None:
        return "⚠️ Please transcribe an audio file first."

    question_embedding = embedding_model.encode([question])
    _, indices = chunk_index.search(np.array(question_embedding), k=3)
    context = " ".join([chunks[i] for i in indices[0]])

    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Minimal but modern CSS
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

body, .gradio-container {
    font-family: 'Poppins', sans-serif !important;
    background: #0f172a !important;
    color: #e0e7ff !important;
    margin: 0; padding: 0;
}

h1, h2, h3 {
    color: #e0e7ff !important;
    font-weight: 600 !important;
}

.gr-button {
    background: linear-gradient(135deg, #7c3aed, #4338ca) !important;
    color: white !important;
    border-radius: 20px !important;
    padding: 10px 25px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}

.gr-textbox textarea, .gr-textbox input[type="text"] {
    background: rgba(255,255,255,0.12) !important;
    border: none !important;
    border-radius: 15px !important;
    padding: 10px 18px !important;
    color: #e0e7ff !important;
}

.gr-block {
    border-radius: 20px !important;
    background: rgba(255, 255, 255, 0.05);
    padding: 30px !important;
    max-width: 900px !important;
    margin: 30px auto !important;
}
"""

# Interface
with gr.Blocks(css=custom_css) as app:
    gr.Markdown(
        """
        # 🎧 Audio-Based Q&A System  
        ### Powered by Whisper + MiniLM + DistilBERT  
        <span style="color:#a5b4fc;">Upload an audio file, get a transcript, and ask questions from it!</span>
        """
    )

    with gr.Group():
        with gr.Row():
            audio_input = gr.Audio(type="filepath", label="🎤 Upload Audio")
            transcript_output = gr.Textbox(label="📄 Transcript", lines=10)

        transcribe_btn = gr.Button("🎬 Transcribe")

        with gr.Row():
            question_input = gr.Textbox(label="💬 Ask a Question", placeholder="e.g. What is the topic?")
            answer_output = gr.Textbox(label="✅ Answer")

        query_btn = gr.Button("🔍 Get Answer")

    transcribe_btn.click(fn=transcribe_audio, inputs=audio_input, outputs=transcript_output)
    query_btn.click(fn=answer_question, inputs=question_input, outputs=answer_output)

port = int(os.environ.get("PORT", 7860))
app.launch(server_name="0.0.0.0", server_port=port)

