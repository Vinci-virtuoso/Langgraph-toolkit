from app.database import db
from app.logic.retrieval import create_retriever

async def retrieve_node(state):
    # Convert state to a normal dictionary so we can add new keys.
    state = dict(state)
    print("State in retrieval node:", state)
    retriever = create_retriever(db, k=2)
    
    # Retrieve the 'question' field, or try to extract it from 'messages' if missing.
    question = state.get("question")
    if not question:
        if "messages" in state and state["messages"]:
            last_message = state["messages"][-1]
            if isinstance(last_message, dict):
                question = last_message.get("content", "")
            else:
                question = getattr(last_message, "content", "")
            # If the content is a list (e.g., from a HumanMessage), extract the text from the first item.
            if isinstance(question, list):
                if question and isinstance(question[0], dict) and "text" in question[0]:
                    question = question[0]["text"]
                else:
                    question = str(question)
        else:
            question = ""
        state["question"] = question
     
    # Invoke the retriever only once with a guaranteed string.
    state["context"] = await retriever.ainvoke(state.get("question", ""))
    return state