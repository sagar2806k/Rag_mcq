import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema.runnable import RunnableSequence
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
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

def create_mcq_chain():
    # Single MCQ Generation Prompt
    mcq_prompt = PromptTemplate(
        input_variables=["context", "topic", "num_questions"],  # Only these variables will be used
        template="""
        Based on the following context, generate {num_questions} multiple choice questions about {topic}.
        
        Context: {context}
        
        Requirements for MCQs:
        1. Each question should be based on the context provided
        2. Each question must have exactly 4 options (A, B, C, D)
        3. Only one option should be correct
        4. Questions should test understanding, not just memory
        
        Format each MCQ as:
        Q1. [Question text]
        A) [Option]
        B) [Option]
        C) [Option]
        D) [Option]
        Correct Answer: [A/B/C/D]

        [Repeat this format for all {num_questions} questions]
        """
    )
    
    # Create LLM
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    
    # Create chain using pipe operator
    chain = mcq_prompt | llm
    return chain
    

def create_question_paper_chain():
    question_paper_prompt = PromptTemplate(
        input_variables=["context", "topic", "total_marks"],
        template="""
        Generate a question paper based on this context about: {topic}
        Total Marks: {total_marks}
        
        Context: {context}
        
        Rules for Question Paper:
        1. Questions should be strictly based on the provided context
        2. Each question should test different concepts
        3. Questions should be clear and specific
        
        For {total_marks} marks paper:
        - If 20 marks:
          * 4 short questions (2 marks each) = 8 marks
          * 3 medium questions (4 marks each) = 12 marks
        
        - If 30 marks:
          * 5 short questions (2 marks each) = 10 marks
          * 4 medium questions (3 marks each) = 12 marks
          * 2 long questions (4 marks each) = 8 marks
        
        Format:
        Question Paper: {topic}
        Total Marks: {total_marks}
        Time: 60 minutes for 20 marks, 90 minutes for 30 marks
        
        Instructions:
        1. Attempt all questions
        2. Write answers to the point
        3. Marks are indicated against each question
        
        Questions:
        [Generate questions following the marks distribution above]
        
        Format each question as:
        Q[number]) [Question] {{marks}} Marks
        """
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    chain = question_paper_prompt | llm
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def generate_mcqs(topic, num_questions):
    try:
        with st.spinner(f"Generating {num_questions} MCQs about {topic}..."):
            # Get embeddings and load vector store
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            
            # Get relevant documents
            docs = new_db.similarity_search(topic, k=3)
            context = " ".join([doc.page_content for doc in docs])
            
            # Generate MCQs
            chain = create_mcq_chain()
            response = chain.invoke({
                "context": context,
                "topic": topic,
                "num_questions": num_questions
            })
            
            # Display results
            st.subheader(f"MCQs for: {topic}")
            st.write(response)
            
    except Exception as e:
        st.error(f"Error generating MCQs: {str(e)}")

def generate_question_paper(topic, total_marks):
    try:
        with st.spinner(f"Generating {total_marks} marks question paper for {topic}..."):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            
            docs = new_db.similarity_search(topic, k=3)
            context = " ".join([doc.page_content for doc in docs])
            
            chain = create_question_paper_chain()
            response = chain.invoke({
                "context": context,
                "topic": topic,
                "total_marks": total_marks
            })
            
            st.subheader(f"Question Paper: {topic} ({total_marks} Marks)")
            st.write(response)
            
            st.download_button(
                label="Download Question Paper",
                data=response,
                file_name=f"question_paper_{topic}_{total_marks}marks.txt",
                mime="text/plain"
            )
            
    except Exception as e:
        st.error(f"Error generating question paper: {str(e)}")

def main():
    st.set_page_config("Education AI Assistant")
    st.header("PDF Learning Assistant")

    tab1, tab2, tab3 = st.tabs(["Chat with PDF", "Generate MCQs", "Generate Question Paper"])
    
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

    with tab3:
        st.subheader("Question Paper Generator")
        qp_topic = st.text_input("Enter topic for Question Paper")
        marks = st.radio(
            "Select Total Marks",
            options=[20, 30],
            horizontal=True
        )
        
        if st.button("Generate Question Paper"):
            if not qp_topic:
                st.warning("Please enter a topic first!")
            else:
                generate_question_paper(qp_topic, marks)

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