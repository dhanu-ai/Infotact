from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import shutil
import tempfile
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Set up Gemini API key
gemini_api_key = os.getenv("GOOGLE_API_KEY")
if gemini_api_key is None:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=gemini_api_key)

# In-memory storage
pdf_raw_text_storage = {}
vector_store_storage = {}
chain_storage = {}

# Core Functions
def get_pdf_text_from_path(file_path):
    text = ""
    pdf_reader = PdfReader(file_path)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(raw_text)

def get_vector(chunks):
    if not chunks:
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=gemini_api_key)
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

def conversation_chain():
    template = """
    You are an expert document summarizer.
    Summarize the following document clearly and concisely.
    Focus on the main ideas, key points, and important facts.

    Context: \n{context}\n
    Question: \n{question}\n
    Answer:
    """
    model_instance = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=gemini_api_key)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return load_qa_chain(model_instance, chain_type="stuff", prompt=prompt)

def user_question(question, db, chain, raw_text):
    if db is None:
        return "No vector database found. Please upload and process a PDF first."
    
    docs = db.similarity_search(question, k=5)
    response = chain.invoke(
        {"input_documents": docs, "question": question, "context": raw_text},
        return_only_outputs=True
    )
    return response.get("output_text")

# API Endpoints

@app.post("/upload_and_summarize_pdf")
async def upload_and_summarize_pdf(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        # Process the file
        raw_text = get_pdf_text_from_path(tmp_path)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="Uploaded PDF is empty or unreadable.")

        chunks = get_text_chunks(raw_text)
        vector_store = get_vector(chunks)
        chain = conversation_chain()

        # Store in memory
        session_id = file.filename  # simple session key
        pdf_raw_text_storage[session_id] = raw_text
        vector_store_storage[session_id] = vector_store
        chain_storage[session_id] = chain

        # Generate summary immediately
        summary_question = "Summarize the uploaded document in a clear and concise way."
        summary = user_question(summary_question, vector_store, chain, raw_text)

        return JSONResponse({
            "message": "PDF uploaded and summarized successfully.",
            "session_id": session_id,
            "summary": summary
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
