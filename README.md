# RAG Project

# Groq_RAG

### streamlit architecture
```text
┌───────────────────────┐
│      Streamlit UI     │
│ ┌───────────────────┐ │
│ │ Sidebar            │ │
│ │ - Upload PDF       │ │
│ │ - Select Document  │ │
│ └───────────────────┘ │
│ ┌───────────────────┐ │
│ │ Main Area          │ │
│ │ - Query Input      │ │
│ │ - Answer Display   │ │
│ └───────────────────┘ │
└─────────────┬─────────┘
              │
              ▼
       ┌───────────────┐
       │ Session State │
       │ - db          │
       │ - selected_doc│
       └───────────────┘
              │
              ▼
       ┌───────────────┐
       │ PDF Processing│
       │ - PyPDFLoader │
       │ - Text Split  │
       └───────────────┘
              │
              ▼
       ┌───────────────┐
       │ Vector Store  │
       │   (Chroma)    │
       │ - store embeddings
       │ - persist data
       └───────────────┘
              │
              ▼
       ┌───────────────┐
       │ Retrieval     │
       │ - As Retriever│
       │ - Fetch relevant chunks
       └───────────────┘
              │
              ▼
       ┌───────────────┐
       │ RAG Chain     │
       │ - Stuff Chain │
       │ - Groq LLM    │
       │ - Cohere Embeds
       └───────────────┘
              │
              ▼
       ┌───────────────┐
       │  Response     │
       │ - Answer      │
       │ - Reasoning   │
       └───────────────┘
```

# RAG pipeline

```text
          ┌───────────────┐
          │   User Query  │
          └──────┬────────┘
                 │
                 ▼
          ┌───────────────┐
          │  Retriever    │
          │ - Fetch top-k │
          │   relevant    │
          │   document    │
          └──────┬────────┘
                 │
                 ▼
          ┌───────────────┐
          │  Retrieved    │
          │  Chunks/Docs  │
          └──────┬────────┘
                 │
                 ▼
          ┌───────────────┐
          │  LLM / Chain  │
          │ - Stuff Chain │
          │ - System Prompt│
          │ - Generates   │
          │   Answer      │
          └──────┬────────┘
                 │
                 ▼
          ┌───────────────┐
          │   Response    │
          │ - Detailed    │
          │   Answer      │
          └───────────────┘
```