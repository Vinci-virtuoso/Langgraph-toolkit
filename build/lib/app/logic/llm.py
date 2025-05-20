from langchain_core.tools import tool
from langchain_groq import ChatGroq


@tool
def get_weather(location: str):
    """Call to get the current weather."""
    if location.lower() in ["munich"]:
        return "It's 15 degrees Celsius and cloudy."
    else:
        return "It's 32 degrees Celsius and sunny."

def create_llm(model_name: str = "llama3-8b-8192") -> ChatGroq:
    """
    Factory function to create and configure the LLM model with the provided tools.

    Args:
        model_name (str): The name of the language model to use.

    Returns:
        ChatOpenAI: The configured LLM instance.
    """
    tools = [get_weather]
    model = ChatGroq(model=model_name)
    model = model.bind_tools(tools)
    return model
