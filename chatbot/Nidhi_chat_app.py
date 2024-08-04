# pip install streamlit pypdf langchain langchain-community langchain-core langchain-text-splitters langsmith faiss-cpu tiktoken langchain_cohere
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_cohere.chat_models import ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os

os.environ['COHERE_API_KEY'] = 'G7cUblbv0CIOy9WK57xRNTs7vXYRA0.......'

st.set_page_config(layout='wide', page_title='The Nidhi Query Bot 😊')

st.title("Welcome to Our App ! 🙏")

pdf_files = st.file_uploader(accept_multiple_files=True, type=['pdf'], label="Upload the pdf here")

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=0
)

if pdf_files:
    all_docs = []
    for pdf in pdf_files:
        # Save the uploaded file temporarily
        with open(pdf.name, "wb") as f:
            f.write(pdf.getbuffer())

        # Load the PDF using PyPDFLoader
        doc = PyPDFLoader(pdf.name).load()
        all_docs.extend(doc)

        # Clean up the temporary file
        os.remove(pdf.name)

    all_texts = " ".join(text.page_content for text in all_docs)

    split_docs = splitter.split_text(all_texts)

    embd = CohereEmbeddings()
    chat = ChatCohere()

    db = FAISS.from_documents(all_docs, embedding=embd)
    retriever = db.as_retriever()

    template = """
        Your job is to provide the correct answer to a given query from the given context. I'll provide you the context along with the query. 
        If the answer is present in the context, then give the answer; otherwise, say, "The provided context doesn't contain enough information to answer this question".
        Here is the context and query:
        context: {context}\n
        query: {query}
    """

    prompt = ChatPromptTemplate.from_template(template)

    query = st.text_area(label="Enter your query below 👇")
    chain = {"query":RunnablePassthrough(), "context": retriever} | prompt | chat
    if st.button("Get Answer"):
        try:
            response = chain.invoke(query)
            st.write(response.content)
        except Exception as e:
            st.write(e)
