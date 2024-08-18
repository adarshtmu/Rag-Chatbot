import streamlit as st
import os
import glob
from typing import Union
from io import BytesIO
from typing import List
from dotenv import load_dotenv
from multiprocessing import Pool
from constants import CHROMA_SETTINGS
import tempfile
from tqdm import tqdm
import argparse
import time
from PIL import Image
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS,Chroma
from langchain_community.llms import Ollama
from langchain_cohere import CohereEmbeddings




load_dotenv()


######################### HTML CSS ############################
css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''
 
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.pinimg.com/originals/0c/67/5a/0c675a8e1061478d2b7b21b330093444.gif" style="max-height: 70px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''
 
 
user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://th.bing.com/th/id/OIP.uDqZFTOXkEWF9PPDHLCntAHaHa?pid=ImgDet&rs=1" style="max-height: 80px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
###################################################
 
chunk_size = 500
chunk_overlap = 50
persist_directory = os.environ.get('PERSIST_DIRECTORY')
print(persist_directory)
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
target_source_chunks= int(os.environ.get('TARGET_SOURCE_CHUNKS', 5))
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
model_type=os.environ.get('MODEL_TYPE')

 
from langchain_community.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader)
 
 
# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}
 
 
 
 
 
 
def load_single_document(file_content: BytesIO, file_type:str) -> List[Document]:
    ext = "." + file_type.rsplit("/", 1)[1]
 
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
        temp_file.write(file_content.getvalue())
        temp_file_path = temp_file.name
 
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(temp_file_path, **loader_args)
        results = loader.load()
        os.remove(temp_file_path)
        return results
 
    raise ValueError(f"Unsupported file extension '{ext}'")
 
 
        
def load_uploaded_documents(uploaded_files, uploaded_files_type, ignored_files: List[str] = []) -> List[Document]:
    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(uploaded_files), desc='Loading new documents', ncols=80) as progress_bar:
            for i, uploaded_file in enumerate(uploaded_files):
                file_type = uploaded_files_type[i]
                file_content=BytesIO(uploaded_file.read())
                docs = load_single_document(file_content, file_type)
                results.extend(docs)
                progress_bar.update()
    return results
 
 
def get_pdf_text(uploaded_files):
    ignored_files = []  # Add files to ignore if needed
 
    uploaded_files_list = [file for file in uploaded_files]
    uploaded_files_type = [file.type for file in uploaded_files]
    results = load_uploaded_documents(uploaded_files_list, uploaded_files_type, ignored_files)
    return results
 
 
 
 
def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 1 documents are needed in a working vectorstore
            if len(list_index_files) > 0:
                print("Yes vectorstore exists")
                return True
    return False
 
 
 
def get_text_chunks(results,chunk_size,chunk_overlap):
    texts=[]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(results)                   
    return texts
 
 
def get_vectorstore(results,embeddings_model_name,persist_directory,client_settings,chunk_size,chunk_overlap):
    if embeddings_model_name == "openai":     
        embeddings = OpenAIEmbeddings()
        print('OpenAI embeddings loaded')
    elif embeddings_model_name == "Cohereembeddings":
        embeddings = CohereEmbeddings(model="embed-english-v3.0")
        print('Cohere embeddings loaded')
       
    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        #print(f"Creating embeddings. May take some minutes...")
        # #print(f"Loaded text size:{len(texts)}")
        texts=get_text_chunks(results,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        if len(texts)>0:
            db.add_documents(texts)  
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        print(f"Creating embeddings. May take some minutes...")
        texts=get_text_chunks(results,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
        db.add_documents(texts)
        
    return db
 

def get_conversation_chain(vectorstore,target_source_chunks,model_type):
    retriever = vectorstore.as_retriever(search_kwargs={"k": target_source_chunks})
    
    # activate/deactivate the streaming StdOut callback for LLMs
    #callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM.
                        
    match model_type:
        case "OpenaAI":
            llm= ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        case "Llama3":
            llm = Ollama(model="llama3")
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: ")
    
    
    #llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    # llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, verbose=False)
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return conversation_chain


st.set_page_config(page_title="Generate Insights",page_icon=":bar_chart:")
 

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
 
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
 


 

def add_logo(logo_path, width, height):
        """Read and return a resized logo"""
        logo = Image.open(logo_path)
        modified_logo = logo.resize((width, height))
        return modified_logo
 
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
col1, col2,col3,col4,col5,col6 = st.columns(6)

#with col5:
                #my_logo = add_logo(logo_path="CampusX.jfif", width=100, height=20)
                #st.image(my_logo)
#with col6:
                #pg_logo=add_logo(logo_path="Q&A logo.jfif", width=60, height=40)
                #st.image(pg_logo)
 
 
 


def main():
    load_dotenv()
    css2 = '''
    <style>
        [data-testid="stSidebar"]{
            min-width: 300px;
            max-width: 300px;
        }
    </style>
    '''
    st.markdown(css2, unsafe_allow_html=True)
 
    st.write(css, unsafe_allow_html=True)
 
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
 
    st.header(":blue Generate Insights :bar_chart:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
 
    with st.sidebar:
        st.subheader("Your documents")
        uploaded_files = st.file_uploader("Upload documents", type=["pdf", "xlsx",'csv'], accept_multiple_files=True)
        #texts=[]
        
        if st.button("Process"):
            with st.spinner("Processing"):
                
                # get pdf text
                if uploaded_files is not None :
                    raw_text = get_pdf_text(uploaded_files=uploaded_files)
 
                    # get the text chunks
                    text_chunks = get_text_chunks(results=raw_text,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
 
                    # create vector store
                    vectorstore = get_vectorstore(results=text_chunks,embeddings_model_name=embeddings_model_name,persist_directory=persist_directory,client_settings=CHROMA_SETTINGS,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
 
                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore=vectorstore,target_source_chunks=target_source_chunks,model_type=model_type)
 
 
if __name__ == '__main__':

    main()

 
