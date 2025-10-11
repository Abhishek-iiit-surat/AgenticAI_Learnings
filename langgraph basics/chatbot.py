from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import os
from IPython.display import display, Image
from dotenv import load_dotenv
## reducers
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
# # defining the state schema


# setting up the environment variables
load_dotenv()  
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")


class State(TypedDict):
    messages:Annotated[list[dict], add_messages]  # list of messages in the chat history

# defining the functions

def superbot(state:State):
    llm = ChatOpenAI(model_name="gpt-5-nano",temperature=0.7)
    return {"messages":[llm.invoke(state["messages"])]}

#definig the graph

graph = StateGraph(State)
graph.add_node("Superbot", superbot)
graph.add_edge(START, "Superbot")
graph.add_edge("Superbot", END)

graph_builder = graph.compile()

# visulaize the graph

# png_bytes = graph_builder.get_graph().draw_mermaid_png()

# # Save the bytes to a file
# with open("graph.png", "wb") as f:
#     f.write(png_bytes)

print(graph_builder.invoke({'messages':"Hi my name is Abhishek!!"}))