import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import base64

# Sidebar contents
with st.sidebar:
    st.title("ScienceSummary.AI 🚀🪐")
    st.markdown(
        """
        ## About
        This app is a GPT-powered chatbot built to help people analyze scientific papers.
        
        ## How to use:
        - Import your PDF paper from local computer by pressing the browse files button.
        - After the model gives a summary you can ask questions about the paper

        ## OpenAI Key
        - In order to use this demo you must paste your openai api key below. This allows you to pay (a very cheap amount) for what you use.
        - If you need help getting your API key [click here](https://www.youtube.com/watch?v=nafDyRsVnXU)
        """
    )
    openai_api_key = st.text_input("OpenAI API Key", type="password")

def pdf_to_base64(pdf_content):
    """Convert PDF bytes to base64."""
    pdf_base64 = base64.b64encode(pdf_content).decode()
    return f"data:application/pdf;base64,{pdf_base64}"

def main():
    st.header("Welcome to ScienceSummary.AI 🚀🪐")
    load_dotenv()

    # Create columns: main content, spacer, and PDF display
    col1, spacer, col2 = st.columns([9, 1, 12])

    # Upload file in the left column with a unique key
    pdf = col1.file_uploader("Upload PDF Contract", type="pdf", key="unique_pdf_upload_key")

    if pdf is not None:
        # Convert the PDF bytes to a base64 encoded string and display in iframe
        pdf_base64 = pdf_to_base64(pdf.getvalue())
        col2.markdown(
            f'<iframe src="{pdf_base64}" width="100%" height="600px"></iframe>',
            unsafe_allow_html=True,
        )

        with st.spinner("Loading PDF..."):
            pdf_reader = PdfReader(pdf)
            col1.write(pdf_reader)

            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

        with st.spinner("Processing text..."):
            # Split docs into fragments
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            # Embeddings
            store_name = pdf.name[:-4]

            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
                col1.write("Embeddings Loaded from the Disk")
            else:
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)

        # Embed the default prompt for a summary
        default_prompt = (
            "Summarize the scientific paper by providing the main objectives, background, problem statement, purpose, methodology, "
            "key findings, data presentation, statistical significance, interpretation of results, implications, limitations, future research directions, "
            "main takeaways, recommendations, key figures and tables, previous research, research gap, broader impact, and significance, "
            "all in one concise paragraph. "
            "Highlight any unclear or vague language, methodological limitations, key results, data interpretation, study implications, key conclusions, "
            "important figures, table contents, key studies, research gaps, implications, significance, and impact with short bullet points. "
            "Keep the summary and concerns very brief."
            "Make sure to keep this brief, about 4-5 sentences."
        )


        docs = VectorStore.similarity_search(query=default_prompt, k=3)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=default_prompt)
        col1.write(f"Summary: {response}")

        # UI for additional user questions with a unique key
        query = col1.text_input("Ask any questions you have about the paper", key="unique_query_input_key")

        if query:
            with st.spinner("Generating response..."):
                docs = VectorStore.similarity_search(query=query, k=3)
                response = chain.run(input_documents=docs, question=query)
                col1.markdown(f"**Response:** {response}")

    else:
        # If no PDF is loaded, show a message in the right column
        col2.write("Contract Not Loaded")

if __name__ == "__main__":
    if openai_api_key and openai_api_key.startswith("sk-"):
        main()
    else:
        st.warning("Please Enter OpenAI API Key")
