import logging
from livekit.agents import Agent, AgentSession
from livekit.plugins import deepgram, openai, silero
from voice_agent.langgraph_adapter import LangGraphAdapter
from src.app.workflows.rag_workflow import create_rag_graph
from langgraph.pregel.remote import RemoteGraph

logger = logging.getLogger(__name__)

def setup_voice_agent(thread_id: str):
    """
    Set up the voice agent using the updated LiveKit SDK.
    Returns a tuple: (agent, session)
    """
    # Compile the LangGraph workflow from rag_workflow.py
    compiled_graph = create_rag_graph()
    graph = RemoteGraph("agent", url="http://localhost:2024")
    # Wrap the compiled graph with LangGraphAdapter, passing configuration as needed
    adapter = LangGraphAdapter(compiled_graph, config={"configurable": {"thread_id": thread_id}})
    
    # Create an AgentSession using the updated providers
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=adapter,
        tts=openai.TTS(),
    )
    
    # Create an Agent with clear instructions
    agent = Agent(
        instructions="You are a voice assistant designed for clear, concise spoken responses.",
    )
    
    return agent, session

if __name__ == "__main__":
    # Example usage for testing
    thread_id = "example-thread-id"
    agent, session = setup_voice_agent(thread_id)
    logger.info("Voice agent is set up. Connect it to a LiveKit room for testing.")