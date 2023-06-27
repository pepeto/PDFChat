from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import pickle
from pathlib import Path
from dotenv import load_dotenv
import os
import streamlit as st
from streamlit_chat import message
import io
import asyncio

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# vectors = getDocEmbeds("gpt4.pdf")
# qa = ChatVectorDBChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo"), vectors, return_source_documents=True)


async def main():

    async def storeDocEmbeds(file, filename):

        reader = PdfReader(file)
        corpus = ''.join([p.extract_text() for p in reader.pages if p.extract_text()])

        splitter =  RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,)
        chunks = splitter.split_text(corpus)

        embeddings = OpenAIEmbeddings(openai_api_key = api_key)
        vectors = FAISS.from_texts(chunks, embeddings)

        with open(filename + ".pkl", "wb") as f:
            pickle.dump(vectors, f)


    async def getDocEmbeds(file, filename):

        if not os.path.isfile(filename + ".pkl"):
            await storeDocEmbeds(file, filename)

        with open(filename + ".pkl", "rb") as f:
            global vectores
            vectors = pickle.load(f)

        return vectors


    async def conversational_chat(query):
        result = qa({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        # print("Log: ")
        # print(st.session_state['history'])
        return result["answer"]


    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")

    if 'history' not in st.session_state:
        st.session_state['history'] = []


    #Creating the chatbot interface
    # Inicializa 'selected_file' en el estado de la sesión
    if 'selected_file' not in st.session_state:
        st.session_state['selected_file'] = None

    # Inicializa 'file_source' en el estado de la sesión
    if 'file_source' not in st.session_state:
        st.session_state['file_source'] = 'upload'

    st.title("Health Document Chat :")

    if 'ready' not in st.session_state:
        st.session_state['ready'] = False

    # Deja que el usuario elija entre cargar un nuevo archivo o seleccionar uno existente
    file_source = st.radio('Seleccione la fuente del archivo:', ('Subir archivo', 'Seleccionar archivo existente'))

    if file_source == 'Subir archivo':
        # Si el usuario elige cargar un nuevo archivo, muestra el cargador de archivos
        uploaded_file = st.file_uploader("Elija un archivo", type=["pdf", "pkl"])

        if uploaded_file is not None:
            st.session_state['selected_file'] = uploaded_file
            st.session_state['file_source'] = 'upload'

    elif file_source == 'Seleccionar archivo existente':
        # Si el usuario elige seleccionar un archivo existente, muestra una lista desplegable con los archivos '.pkl' existentes
        existing_files = [str(f) for f in Path('.').glob('*.pkl')]
        selected_file = st.selectbox('Seleccione un archivo existente:', existing_files)

        if selected_file:
            st.session_state['selected_file'] = selected_file
            st.session_state['file_source'] = 'select'

    if st.session_state['selected_file'] is not None:
        with st.spinner("Procesando..."):
            if st.session_state['file_source'] == 'upload':
                # Si se ha cargado un nuevo archivo, procede como antes
                uploaded_file = st.session_state['selected_file']
                uploaded_file.seek(0)
                file = uploaded_file.read()
                vectors = await getDocEmbeds(io.BytesIO(file), uploaded_file.name)

            elif st.session_state['file_source'] == 'select':
                # Si se ha seleccionado un archivo existente, solo carga los vectores desde el archivo
                with open(st.session_state['selected_file'], "rb") as f:
                    vectors = pickle.load(f)

            qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo"),
                                                       retriever=vectors.as_retriever(), return_source_documents=True)
        st.session_state['ready'] = True

    st.divider()

    if st.session_state['ready']:

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Puedes preguntar lo que quieras en el idioma que quieras sobre "]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hola!"]

        # container for chat history
        response_container = st.container()

        # container for text box
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Pregunta:", placeholder="x ej: Resume el paper en unas pocas frases", key='input')
                submit_button = st.form_submit_button(label='Enviar')

            if submit_button and user_input:
                output = await conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")


if __name__ == "__main__":
    asyncio.run(main())