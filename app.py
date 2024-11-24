print("sagar you can do it man....")
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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

def get_conversational_chain():
    prompt_template = """ 
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
    return chain

def get_mcq_chains():
    # Content Selection Chain
    content_prompt = PromptTemplate(
        input_variables=["context", "topic"],
        template="""
        From the following context, extract information relevant to the topic: {topic}
        
        Context: {context}
        
        Relevant Information:"""
    )

    content_chain = LLMChain(
        llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3),
        prompt=content_prompt,
        output_key="relevant_content"
    )

    # MCQ Generation Chain
    mcq_prompt = PromptTemplate(
        input_variables=["relevant_content", "num_questions"],
        template="""
        Generate {num_questions} multiple choice questions based on this content:
        {relevant_content}
        
        For each question:
        1. Create a clear, specific question
        2. Provide 4 options (A, B, C, D)
        3. Ensure only one correct answer
        4. Mark the correct answer
        
        Format each MCQ as:
        Q[number]. [Question]
        A) [Option]
        B) [Option]
        C) [Option]
        D) [Option]
        Correct: [A/B/C/D]
        
        Generate MCQs:"""
    )
    
    mcq_chain = LLMChain(
        llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7),
        prompt=mcq_prompt,
        output_key="mcqs"
    )

    # Combine chains
    sequential_chain = SequentialChain(
        chains=[content_chain, mcq_chain],
        input_variables=["context", "topic", "num_questions"],
        output_variables=["relevant_content", "mcqs"]
    )
    
    return sequential_chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def generate_mcqs(topic, num_questions):
    try:
        with st.spinner(f"Generating {num_questions} MCQs about {topic}..."):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            
            # Get relevant documents
            docs = new_db.similarity_search(topic, k=3)
            context = " ".join([doc.page_content for doc in docs])
            
            # Generate MCQs using sequential chain
            mcq_chain = get_mcq_chains()
            response = mcq_chain({
                "context": context,
                "topic": topic,
                "num_questions": num_questions
            })
            
            # Display MCQs
            st.subheader(f"MCQs for: {topic}")
            st.write(response["mcqs"])
            
    except Exception as e:
        st.error(f"Error generating MCQs: {str(e)}")


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF & MCQ Generator")

    tab1, tab2 = st.tabs(["Chat with PDF", "Generate MCQs"])
    
    with tab1:
        user_question = st.text_input("Ask a question from the PDF Files..")
        if user_question:
            user_input(user_question)

    with tab2:
        st.subheader("MCQ Generator")
        mcq_topic = st.text_input("Enter the topic or concept for MCQs")
        num_questions = st.number_input(
            "Number of MCQs to generate",
            min_value=5,
            max_value=30,
            value=10,
            step=5
        )
        
        if st.button("Generate MCQs"):
            if not mcq_topic:
                st.warning("Please enter a topic first!")
            else:
                generate_mcqs(mcq_topic, num_questions)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()