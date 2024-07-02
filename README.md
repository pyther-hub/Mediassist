# RAG Medical Assistant Project

### Project Description:

The RAG (Medical Assistant) project is designed to assist in medical information retrieval and knowledge management. It utilizes various scripts and notebooks to collect, process, and utilize medical data for building a knowledge graph and assisting with medical queries.

### Tech Stack:

* **OpenAI GPT-3.5**: Used for creating the knowledge graph from TogetherAI.
* **LLaMA3 Model**: Utilized for inference tasks related to processing the knowledge graph.
* **Neo4j**: Database management system used for storing and querying the knowledge graph.
* **LangChain**: Framework used in the overall pipeline for handling natural language processing tasks.


### Project Files:

*   **advanced\_rag.py**: Main script for advanced functionalities of the RAG assistant.
*   **colletedata\_wikipedia.ipynb**: Jupyter notebook for collecting medical data from Wikipedia.
*   **common\_diseases.txt**: Text file listing common diseases for reference or data processing.
*   **create\_knowledge\_graph.ipynb**: Jupyter notebook for creating and visualizing the knowledge graph.
*   **graph\_documents.pkl**: Pickle file storing documents related to the knowledge graph.
*   **kg\_helper.py**: Helper functions and utilities for managing the knowledge graph.
*   **llm\_model.py**: Script or module for handling natural language processing tasks with language models.
*   **prompt\_templates.py**: Templates and prompts for interacting with the assistant.
*   **requirements.txt**: File specifying Python dependencies required for the project.
*   **test.ipynb**: Jupyter notebook for testing functionalities and modules within the project.

This project integrates data collection, knowledge graph creation, and natural language processing to provide a comprehensive medical assistant tool.
