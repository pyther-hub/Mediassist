
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser, StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

template = """
Extract all entities related to persons, diseases, medical conditions, symptoms, and diagnoses from the given USER_QUERY. 
{format_instructions}

USER_QUERY:
{user_query}
"""
parser = CommaSeparatedListOutputParser()


extract_entities_parser = CommaSeparatedListOutputParser()
extract_entities_prompt = PromptTemplate(
    input_variables=['user_query'],
    template = template,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

rag_template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)