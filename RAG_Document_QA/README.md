# RAG Document Q&A System

A beginner-friendly Retrieval-Augmented Generation application developed in Python for the Big Brains Generative AI Internship.

## Features

- Reads a custom text knowledge base
- Splits documents into meaningful chunks
- Generates embeddings with Sentence Transformers
- Stores and searches document vectors using ChromaDB
- Uses retrieved context to generate answers through the Gemini API
- Tests questions with different wording and questions not covered by the knowledge base

## Technologies

- Python
- Sentence Transformers
- ChromaDB
- Google Gemini API

## Project Files

- `rag_app.py`: Main application
- `knowledge_base.txt`: Sample university knowledge base
- `requirements.txt`: Python dependencies
- `.gitignore`: Excludes local and sensitive files

## Setup

1. Install Python.
2. Create and activate a virtual environment.
3. Install dependencies:

   `python -m pip install -r requirements.txt`

4. Configure the `GEMINI_API_KEY` environment variable.
5. Run the application:

   `python rag_app.py`

## How It Works

The application loads the knowledge base, divides it into chunks, generates embeddings, and stores the vectors in ChromaDB. When a user asks a question, the system retrieves relevant chunks and provides them to Gemini as context for generating an answer.

## Testing

Test direct questions, differently worded questions, and unsupported questions. Compare the answers with the original knowledge base.

## Note

Nova University and all associated information are fictional sample data for educational use.