from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from typing import Any

def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text
    search. It processes the input string by splitting it into words and 
    appending a similarity threshold (~2 changed characters) to each
    word, then combines them using the AND operator. Useful for mapping
    entities from user questions to database values, and allows for some 
    misspellings.

    Args:
        input (str): The input string to generate a full-text query for.

    Returns:
        str: A full-text search query string.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

def structured_retriever(graph: Any, structured_llm: Any, question: str) -> str:
    """
    Retrieves structured data based on entities identified in the user question.

    This function queries a graph database to retrieve structured data related
    to entities identified in the user question processed by a structured language model.

    Args:
        graph (Any): The graph database or similar structure to query.
        structured_llm (Any): Structured language model or processor to extract entities.
        question (str): The user question to retrieve structured data for.

    Returns:
        str: Structured data retrieved from the graph database.
    """
    result = ""
    entities = structured_llm.invoke({'user_query': question})
    print(entities)
    for entity in entities:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, 
            {limit:1})
            YIELD node,score
            CALL {
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS 
              output
              UNION
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS 
              output
            }
            RETURN output LIMIT 6
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result
