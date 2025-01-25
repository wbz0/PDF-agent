from pdf_loader import process_uploaded_file
from file_splitter import split_documents
from RAGSystem import RAGSystem

def qa_agent(openai_api_key, uploaded_file, question, chat_history):

    """模块一：上传文档"""
    docs = process_uploaded_file(uploaded_file)

    """模块二：文档分割"""
    texts = split_documents(docs)

    """模块三：向量存储与基于检索的对话系统实现"""
    System = RAGSystem(openai_api_key, texts, question, chat_history)

    response = System.qa()
    return response["answer"]
