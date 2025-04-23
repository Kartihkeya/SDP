import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import tempfile
import subprocess
import os

# Set your Google API Key (secure via .env in production)
os.environ["GOOGLE_API_KEY"] = "AIzaSyBmYv0r6YSm1xoBFuI7nVkKp6tcgAl_Yyo"

# Extract text from a single PDF safely
def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        return "".join([page.extract_text() or "" for page in reader.pages])
    except:
        return ""

# Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# Detect suspicious elements using pdfid.py
def is_pdf_suspicious(uploaded_file) -> bool:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(uploaded_file.read())
        tmp_pdf_path = tmp_pdf.name

    result = subprocess.run(
        ["python", "pdfid.py", tmp_pdf_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    output = result.stdout

    suspicious_keywords = ["/JavaScript", "/JS", "/Launch", "/EmbeddedFile", "/OpenAction", "/AA"]
    for line in output.splitlines():
        for keyword in suspicious_keywords:
            if keyword in line:
                try:
                    count = int(line.split(":")[1].strip())
                    if count > 0:
                        return True
                except:
                    continue
    return False

# Create FAISS vector store, with fallback for embedding failures
def get_vector_store(text_chunks):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_texts(text_chunks, embeddings)
        db.save_local("faiss_index")
    except IndexError:
        st.error("⚠️ Potentially malicious or corrupted PDF detected (embedding failure).")
        st.stop()

# Create QA chain using Gemini
def get_conversational_chain():
    prompt = PromptTemplate.from_template("""
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, say, "Answer is not available in the context." 

    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """)
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Handle user question
def user_input(user_question):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    print(response)
    st.write("Reply:", response["output_text"])

# Main app logic
def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Probe Detection Using LLM")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question.strip():
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

        if pdf_docs and st.button("Submit & Process"):
            combined_text = ""

            with st.spinner("Scanning PDFs for malware and readability..."):
                for pdf in pdf_docs:
                    # Malware scan
                    if is_pdf_suspicious(pdf):
                        st.error(f"⚠️ {pdf.name} appears to contain potentially malicious elements.")
                        return

                    # Reset pointer after malware scan
                    pdf.seek(0)

                    # Readability scan
                    text = extract_text_from_pdf(pdf)
                    if not text.strip():
                        st.error(f"⚠️ {pdf.name} has no readable content. It may be corrupted or malicious.")
                        return

                    combined_text += text

                st.success("✅ All files passed malware and readability checks.")

            with st.spinner("Processing..."):
                text_chunks = get_text_chunks(combined_text)

                if not text_chunks:
                    st.error("⚠️ No valid chunks could be generated. Please upload a different PDF.")
                    return

                get_vector_store(text_chunks)
                st.success("✅ Done! You can now ask questions.")

if __name__ == "__main__":
    main()
