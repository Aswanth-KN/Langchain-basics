import os
from dotenv import find_dotenv, load_dotenv
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import (Docx2txtLoader,PyPDFLoader, TextLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_groq import ChatGroq

# loading PDF, DOCX and TXT files as LangChain Documents
def load_documents(file_path):
    name, ext = os.path.splitext(file_path)

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        data =  loader.load()
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
        data =  loader.load()
    elif ext == ".txt":
        loader = TextLoader(file_path)
        data =  loader.load()
    else:
        st.error("Unsupported file format")
        return None
    return data


# splitting data in chunks
def chunk_data(data, chunk_size= 256, chunk_overlap= 20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.create_documents([doc.page_content for doc in data])
    return chunks


# create embeddings using HuggingFaceEmbeddings() and save them in a Chroma vector store
def creating_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def ask_and_get_answer(vector_store, q, k=3):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key= os.getenv("GROQ_API_KEY")
    )

    retrieveer = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retrieveer, chain_type='stuff')

    answer = chain.invoke(q)
    return answer['result']


def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    st.subheader('LLM Question-Answering Application 🤖')

    with st.sidebar:
        st.header("Upload your document here")
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

        chunk_size = st.slider("Chunk Size", min_value=100, max_value=1000, value=256, step=50)
        k = st.slider("Number of similar chunks to retrieve (k)", min_value=1, max_value=10, value=3)
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner('Reading, chunking and embedding file ...'):
                byte_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)

                with open(file_name, 'wb') as f:
                    f.write(byte_data)

                data = load_documents(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f"chunk size: {chunk_size}, number of chunks: {len(chunks)}")

                vector_store = creating_embeddings(chunks)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')
    
    q = st.text_input("Ask a question about the content of your file:")
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            st.write(f'k : {k}')
            answer = ask_and_get_answer(vector_store, q, k=k)
            st.text_area('LLM Answer', value=answer, height=200)
            st.divider()

            if 'history' not in st.session_state:
                st.session_state.history = ''

            value = f"Q: {q}\nA: {answer}\n\n"

            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'

            h = st.session_state.history

            st.text_area('Conversation History', value=h, height=400)

