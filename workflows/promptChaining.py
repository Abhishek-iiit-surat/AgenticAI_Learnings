from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os

# Load environment
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.4)

# Define the state
class State(TypedDict):
    topic: str
    story: str
    improved_story: str
    final_story: str

# Node: generate story
def generate_story(state: State):
    msg = llm.invoke(f"Write one story premise on the topic '{state['topic']}'.")
    return {"story": msg.content}

# Conditional function (NOT a node)
def check_conflict(state: State):
    msg = llm.invoke(
        f"Does the following story have any conflicts or plot holes? "
        f"Story: {state['story']}. Answer with 'yes' or 'no'."
    )
    if "yes" in msg.content.lower():
        return "Fail"
    else:
        return "Pass"

# Node: improve story
def improve_story(state: State):
    msg = llm.invoke(f"Improve the following story by fixing any conflicts or plot holes: {state['story']}")
    return {"improved_story": msg.content}

# Node: finalize story
def finalize_story(state: State):
    text = state.get("improved_story", state["story"])
    msg = llm.invoke(f"Make the following story more engaging and interesting: {text}")
    return {"final_story": msg.content}

# Build the graph
graph = StateGraph(State)

graph.add_node("generate_story", generate_story)
graph.add_node("improve_story", improve_story)
graph.add_node("finalize_story", finalize_story)

# Flow:
graph.add_edge(START, "generate_story")

# Conditional branching based on check_conflict
graph.add_conditional_edges(
    "generate_story",  # branching after this node
    check_conflict,    # function returning "Pass" or "Fail"
    {"Pass": "finalize_story", "Fail": "improve_story"}
)

graph.add_edge("improve_story", "finalize_story")
graph.add_edge("finalize_story", END)

graph_builder = graph.compile() 
for chunk in graph_builder.stream( {"topic": "A space adventure"},stream="updates"):
    print(chunk)

