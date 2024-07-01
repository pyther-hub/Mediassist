from typing import Any, Dict, List
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from kg_helper import structured_retriever  

def retriever(graph: Any, vector_index: Any, structured_llm: Any, question: str) -> str:
    """
    Retrieves structured and unstructured data based on a user query.

    Args:
        graph (Any): The knowledge graph or data structure for structured retrieval.
        vector_index (Any): Vector index or similar object for unstructured retrieval.
        structured_llm (Any): Structured language model or similar for structured retrieval.
        question (str): The user query for data retrieval.

    Returns:
        str: A formatted string containing structured and unstructured data.
             Returns None if an error occurs during retrieval.
    """
    try:
        print(f"Search query: {question}")
        structured_data = structured_retriever(graph, structured_llm, question)
        unstructured_data = [
            el.page_content for el in vector_index.similarity_search(question)
        ]
        final_data = f"""You are a helpful Medical assistant and you have to answer user queries from the given context in the form of Structured and Unstructured data.
                        Structured data:
                        {structured_data}
                        Unstructured data:
                        {"#Document ".join(unstructured_data)}
                        """
        return final_data
    except Exception as e:
        print(f"An error occurred while retrieving data: {e}")
        return None

def rerank():
    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever