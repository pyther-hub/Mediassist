from langchain_openai import ChatOpenAI
from langchain_together import Together

class LLM:
    def __init__(self, service: str, model_name: str = None):
        """
        Initialize a Language Model (LLM) instance based on the specified service.

        Args:
            service (str): The service to use ('OpenAI' or 'TogetherAI').
            model_name (str, optional): The model name to use for the LLM instance. Defaults to None.

        Raises:
            ValueError: If an invalid service or model name is provided.
        """
        if service == 'OpenAI':
            if model_name is None:
                model_name = "gpt-3.5-turbo-0125"
            try:
                self.llm = ChatOpenAI(temperature=0, model_name=model_name)
            except KeyError as e:
                raise ValueError(f"Invalid model name '{model_name}' for OpenAI service.") from e
        
        elif service == 'TogetherAI':
            try:
                self.llm = Together(model=model_name)
            except KeyError as e:
                raise ValueError(f"Invalid model name '{model_name}' for TogetherAI service.") from e
        
        else:
            raise ValueError(f"Unsupported service '{service}'. Please choose either 'OpenAI' or 'TogetherAI'.")
