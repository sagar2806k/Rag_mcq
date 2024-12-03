import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pyaudio
import wave
from groq import Groq

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
os.getenv("GROQ_API_KEY")  # Add GROQ_API_KEY to your .env file
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def record_audio(duration=5):
    """Record audio from microphone"""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   frames_per_buffer=CHUNK)
    
    st.info(f"Recording for {duration} seconds...")
    frames = []
    
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save recording
    audio_file = "recording.wav"
    with wave.open(audio_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    return audio_file

def convert_speech_to_text(audio_file):
    """Convert audio to text using Groq API"""
    try:
        with open(audio_file, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_file, file.read()),
                model="whisper-large-v3-turbo",  # Fastest model
                language="en",
                response_format="text"
            )
        # Assuming the response is a string when response_format is "text"
        return transcription
    except Exception as e:
        st.error(f"Error in speech to text conversion: {str(e)}")
        return None

def create_assignment_chain():
    """Create the assignment generation chain"""
    assignment_prompt = PromptTemplate(
        input_variables=["context", "topic"],
        template="""
        Create an assignment based on this topic: {topic}
        
        Context: {context}
        
        Assignment should include:
        1. Clear objectives
        2. Multiple types of questions:
           - Short answer questions (2 marks each)
           - Problem-solving tasks (5 marks each)
           - Analysis questions (8 marks each)
        3. Clear marking scheme
        
        Format:
        Assignment: {topic}
        Duration: 2 hours
        Total Marks: 50
        
        Instructions:
        1. Answer all questions
        2. Write clearly and concisely
        3. Show your work where necessary
        
        Questions:
        [Generate questions with marks distribution]
        """
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    chain = assignment_prompt | llm
    return chain

def generate_assignment(topic):
    """Generate assignment based on topic"""
    try:
        with st.spinner(f"Generating assignment for: {topic}"):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            
            docs = new_db.similarity_search(topic, k=3)
            context = " ".join([doc.page_content for doc in docs])
            
            chain = create_assignment_chain()
            response = chain.invoke({
                "context": context,
                "topic": topic
            })
            
            st.subheader(f"Assignment: {topic}")
            st.write(response)
            
            st.download_button(
                label="Download Assignment",
                data=response,
                file_name=f"assignment_{topic}.txt",
                mime="text/plain"
            )
            
    except Exception as e:
        st.error(f"Error generating assignment: {str(e)}")

def main():
    st.set_page_config("Voice Assignment Generator")
    st.header("Voice-based Assignment Generator")

    with st.sidebar:
        st.title("Upload PDF")
        pdf_docs = st.file_uploader(
            "Upload PDF Files and Click Process",
            accept_multiple_files=True
        )
        if st.button("Process PDFs"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDFs processed successfully!")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Voice Input")
        duration = st.slider("Recording Duration (seconds)", 3, 10, 5)
        
        if st.button("Start Recording"):
            with st.spinner("Recording..."):
                audio_file = record_audio(duration=duration)
                st.audio(audio_file)  # Play back the recording
                
                with st.spinner("Converting speech to text..."):
                    topic = convert_speech_to_text(audio_file)
                    if topic:
                        st.success(f"Detected Topic: {topic}")
                        generate_assignment(topic)
    
    with col2:
        st.subheader("Text Input")
        text_topic = st.text_input("Or type your topic here")
        if st.button("Generate Assignment") and text_topic:
            generate_assignment(text_topic)

if __name__ == "__main__":
    main()