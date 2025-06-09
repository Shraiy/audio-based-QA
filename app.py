import whisper
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import nltk
import os

nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# Load models
asr_model = whisper.load_model("base")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Global variables
chunks = []
chunk_embeddings = None

# Transcription + chunking
def transcribe_audio(audio_file):
    global chunks, chunk_embeddings

    result = asr_model.transcribe(audio_file)
    transcript = result["text"]

    sentences = sent_tokenize(transcript)
    chunk_size = 5
    chunks = [" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    chunk_embeddings = faiss.IndexFlatL2(embeddings.shape[1])
    chunk_embeddings.add(embeddings)

    return transcript

# Question answering
def answer_question(question):
    if not chunks:
        return "\u2757 Please transcribe an audio file first."

    question_embedding = embedding_model.encode([question])
    _, indices = chunk_embeddings.search(np.array(question_embedding), k=3)
    context = " ".join([chunks[i] for i in indices[0]])

    result = qa_pipeline(question=question, context=context)
    return result['answer']


# Custom CSS for modern elegant UI
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

body, .gradio-container {
    font-family: 'Poppins', sans-serif !important;
    background: #0f172a !important;
    color: #e0e7ff !important;
    margin: 0; 
    padding: 0;
    min-height: 100vh;
    overflow-x: hidden;
}

h1, h3 {
    font-weight: 600 !important;
    color: #e0e7ff !important;
    margin: 0;
}

h1 {
    font-size: 2.8rem !important;
    margin-bottom: 0.25rem !important;
    letter-spacing: 1.3px;
}

h3 {
    font-size: 1.1rem !important;
    color: #a5b4fc !important;
    margin-top: 0 !important;
    margin-bottom: 30px !important;
}

.subtitle-text {
    color: #a5b4fc !important;
    font-size: 1.2rem !important;
    font-weight: 500 !important;
    margin-bottom: 40px !important;
}

.gr-block {
    max-width: 900px !important;
    margin: 50px auto !important;
    padding: 40px 50px !important;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 30px !important;
    box-shadow:
        0 4px 30px rgba(99, 102, 241, 0.25),
        inset 0 0 200px rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.15);
}

.gr-button {
    background: linear-gradient(135deg, #7c3aed, #4338ca) !important;
    color: #f0f0ff !important;
    border-radius: 25px !important;
    padding: 15px 40px !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    letter-spacing: 1.1px;
    box-shadow: 0 8px 20px rgba(124, 58, 237, 0.6);
    border: none !important;
    transition: all 0.35s ease-in-out !important;
    user-select: none;
}
.gr-button:hover {
    background: linear-gradient(135deg, #5b21b6, #312e81) !important;
    box-shadow: 0 12px 25px rgba(91, 33, 182, 0.8);
    cursor: pointer;
}

.gr-textbox textarea, .gr-textbox input[type="text"] {
    background: rgba(255, 255, 255, 0.12) !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 18px 25px !important;
    font-size: 1.1rem !important;
    color: #e0e7ff !important;
    box-shadow:
        inset 0 4px 15px rgba(124, 58, 237, 0.25),
        0 0 12px rgba(99, 102, 241, 0.2);
    transition: box-shadow 0.3s ease-in-out !important;
    font-weight: 500 !important;
    user-select: text;
    resize: vertical !important;
    min-height: 60px !important;
}
.gr-textbox textarea:focus, .gr-textbox input[type="text"]:focus {
    outline: none !important;
    box-shadow:
        0 0 15px 2px #7c3aed !important,
        inset 0 0 15px 1px #a78bfa !important;
    background: rgba(255, 255, 255, 0.18) !important;
}

.gr-audio audio {
    border-radius: 20px !important;
    box-shadow: 0 8px 30px rgba(124, 58, 237, 0.7) !important;
    filter: drop-shadow(0 0 15px #7c3aed);
}

.gr-row {
    gap: 30px !important;
    margin-bottom: 35px !important;
}

.gr-row > * {
    flex: 1 1 0 !important;
}

.gradio-container {
    padding-bottom: 80px !important;
}

/* Responsive adjustments */
@media (max-width: 720px) {
    .gr-row {
        flex-direction: column !important;
    }
}
"""

with gr.Blocks(css=custom_css) as app:
    gr.Markdown(
        """
        # 🎧 Audio-Based Q&A System
        ### Built with Whisper, Sentence-BERT, and DistilBERT
        <p class="subtitle-text">Upload an audio file, get the transcript, then ask questions from it!</p>
        """
    )

    with gr.Group(elem_id="main-container"):
        with gr.Row():
            audio_input = gr.Audio(type="filepath", label="🎧 Upload Audio")
            transcript_output = gr.Textbox(label="📝 Transcript", lines=10)

        transcribe_btn = gr.Button("🔀 Transcribe")

        with gr.Row():
            question_input = gr.Textbox(label="❓ Ask a Question", placeholder="What is being said in the audio?")
            answer_output = gr.Textbox(label="🌐 Answer")

        query_btn = gr.Button("🔎 Get Answer")

    transcribe_btn.click(fn=transcribe_audio, inputs=audio_input, outputs=transcript_output)
    query_btn.click(fn=answer_question, inputs=question_input, outputs=answer_output)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port)
