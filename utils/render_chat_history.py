"""
This method will render chats from the provided chat history list
"""

from langchain_core.messages import HumanMessage, AIMessage

def render_chat(chat_context: str, max_turns: int = 8) -> str:
    turns = chat_context[-max_turns:]
    lines = []

    for msg in turns:
        if isinstance(msg, HumanMessage):
            lines.append(f"user: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"assistant: {msg.content}")
    return "\n---\n".join(lines)