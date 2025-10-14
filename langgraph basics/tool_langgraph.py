from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.4)

# Example tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

# Define the graph state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Bind tools to the model
llm_with_tools = llm.bind_tools([add])

# Chatbot node
def chatbot(state: State):
    result = llm_with_tools.invoke(state["messages"])
    print("Chatbot response:", result)
    return {"messages": [result]}

# Build the graph
builder = StateGraph(State)
builder.add_node("Chatbot", chatbot)
builder.add_node("Tools", ToolNode([add]))

# Add edges
builder.add_edge(START, "Chatbot")

# ✅ Conditional routing
builder.add_conditional_edges(
    "Chatbot",
    tools_condition,
    {
        "tools": "Tools",
        "__end__": END,
    },
)

# After tools are executed, return to Chatbot
builder.add_edge("Tools", "Chatbot")

# Compile the graph
graph = builder.compile()

# Example usage
if __name__ == "__main__":
    result = graph.invoke({"messages": ["What is 2 + 3?"]})
    print(result)
