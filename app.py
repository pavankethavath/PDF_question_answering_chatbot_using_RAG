# app.py

import streamlit as st
from rag_model import RAGSystem

def main():
    # Page configuration
    st.set_page_config(
        page_title="PDF Question Answering System",
        page_icon="📚",
        layout="wide"
    )
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = None
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False

    # Main title
    st.title("📚 PDF Question Answering System")

    # Sidebar
    st.sidebar.header("Upload PDF")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

    # Process PDF when uploaded
    if uploaded_file:
        with st.spinner("Processing PDF... This might take a minute..."):
            try:
                success = st.session_state.rag_system.process_pdf(uploaded_file.getvalue())
                if success:
                    st.session_state.query_engine = st.session_state.rag_system.get_query_engine()
                    st.session_state.pdf_processed = True
                    st.sidebar.success("PDF processed successfully!")
                else:
                    st.sidebar.error("Error processing PDF!")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")

    # Main content area
    st.header("Ask a Question")
    question = st.text_input("Enter your question about the PDF content:")

    # Generate response
    if st.button("Get Answer"):
        if not question:
            st.warning("Please enter a question!")
        elif not st.session_state.pdf_processed:
            st.warning("Please upload a PDF first!")
        else:
            with st.spinner("Generating answer..."):
                try:
                    response = st.session_state.rag_system.generate_response(
                        st.session_state.query_engine,
                        question
                    )
                    st.subheader("Answer")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Instructions
    with st.sidebar.expander("ℹ️ Usage Instructions"):
        st.write("""
        1. Upload a PDF file using the uploader above
        2. Wait for the PDF to be processed
        3. Type your question in the main panel
        4. Click 'Get Answer' to generate a response
        5. The system will analyze the PDF content and provide a relevant answer
        """)

if __name__ == "__main__":
    main()