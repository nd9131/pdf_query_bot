### The Nidhi Query Bot

The Nidhi Query Bot(pdf) is a Streamlit-based web application designed to provide intelligent and accurate answers to user queries using context extracted from uploaded PDF documents. The app utilizes advanced natural language processing and machine learning models from the LangChain and Cohere libraries to achieve this. Key features and functionalities include:

- **File Upload**: Users can upload multiple PDF files simultaneously.
- **Text Extraction**: The app extracts and processes text from the uploaded PDFs using the PyPDFLoader.
- **Text Splitting**: The extracted text is split into manageable chunks for efficient processing using the RecursiveCharacterTextSplitter.
- **Embedding and Retrieval**: The extracted text is embedded using CohereEmbeddings, and a FAISS-based vector store is used for efficient text retrieval.
- **Prompt Template**: A custom ChatPromptTemplate guides the AI to answer user queries based on the provided context.
- **Query Handling**: Users can input their queries, and the bot will provide answers based on the context extracted from the PDFs or indicate if the context lacks sufficient information to answer the query.

With a user-friendly interface and powerful backend, the Nidhi Query Bot offers a seamless experience for extracting and querying information from PDF documents.
