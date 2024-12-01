# PDF Question Answering System Using Retrieval-Augmented Generation (RAG)

This project is a sophisticated question-answering system designed to extract and provide context-aware answers from PDF documents. By integrating advanced **Retrieval-Augmented Generation (RAG)** techniques and state-of-the-art AI models, the system enables users to interact with their documents in a more efficient and intelligent manner.

---

## Use Cases

- **Academic Research**: Quickly extract insights from research papers, reports, or studies.
- **Professional Analysis**: Navigate lengthy contracts, whitepapers, or manuals with ease.
- **Everyday Use**: Simplify interactions with dense or complex PDF documents.

---

## Key Features

- **PDF Processing**: Upload and process PDF documents for analysis.
- **Interactive Q&A**: Enter natural-language questions and receive precise answers based on document content.
- **Advanced Retrieval**: Uses vector-based indexing and similarity scoring for accurate content retrieval.
- **User-Friendly Interface**: A web application built with Streamlit ensures ease of use and accessibility.

---

## Technologies Used
Frontend: Streamlit
Backend: Python
Machine Learning:
HuggingFace Transformers for text generation
VectorStoreIndex for document indexing
Custom retriever and postprocessor for improved accuracy

## Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name

2. Run the Application: Start the Streamlit application:
   ```bash
    streamlit run app.py

## Upload a PDF and Start Querying
 ![Home Screen](images/home.png)
- Upload your desired PDF file through the application interface.
 ![Home Screen](images/upload.png)
- Enter questions and retrieve contextually accurate responses.
 ![Home Screen](images/answer.png)

---

## How It Works

1. **PDF Processing**:
   -  The system reads and processes the uploaded PDF, splitting it into manageable chunks for indexing.

3. **Information Retrieval**:
   - The indexed content is retrieved using advanced embeddings and similarity scoring.

4. **Answer Generation**:
   - A pre-trained language model generates context-aware and concise responses based on the retrieved content.

---

## Technology Stack

- **Frontend**: Streamlit for an interactive and intuitive user experience.
- **Backend**:
  - HuggingFace Transformers for natural language understanding and generation.
  - Vector-based retrieval using custom embeddings.
- **Programming Language**: Python.

---

## Code Overview

### `app.py`

- A Streamlit application that provides the user interface.
- Handles PDF uploads, question inputs, and displays answers.

### `rag.py`

- Implements the core RAG logic:
  - **PDF Processing**: Reads and splits the PDF into manageable chunks.
  - **Indexing**: Creates a vector index for efficient content retrieval.
  - **Query Engine**: Uses a retriever and postprocessor to answer queries.
  - **Response Generation**: Generates detailed responses using a transformer model.

---

## Usage Instructions

1. Upload a PDF file.
2. Wait for the system to process the document.
3. Type your question and click "Get Answer".
4. View the answer generated by the system.

---

## Future Enhancements

- **Multi-Document Support**: Enable querying across multiple PDF files.
- **Multi-Language Support**: Add support for processing documents in multiple languages.
- **GPU Support**: Implement GPU acceleration for faster processing and response times.
- **Additional Formats**: Expand support to other document formats such as DOCX and TXT.
- **Enhanced UI**: Improve the user interface with advanced analytics and visualization features.

---


## Contributing

We welcome contributions from the community. To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request detailing your contribution.

For any issues or suggestions, please open a discussion or issue on the repository.

---

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it in compliance with the terms of the license.

---

## Contact

For inquiries or further information, please contact via the repository issue tracker or email (if applicable).

