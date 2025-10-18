from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from fpdf import FPDF
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
import os
import re

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.9)
class Route(BaseModel):
    step: Literal["story", "poem", "joke"] = Field(..., description="Type of creative output to generate")

class State(TypedDict):
    input: str
    decision: str
    output:str

router = llm.with_structured_output(Route)

def poem_generator(state: State):
    msg = llm.invoke(f"Write a poem about: '{state['input']}'")
    return {"output": msg.content}

def joke_generator(state: State):
    msg = llm.invoke(f"Tell a joke about: '{state['input']}'")
    return {"output": msg.content}

def story_generator(state: State):   
    msg = llm.invoke(f"Write a short story about: '{state['input']}'")
    return {"output": msg.content}

def decide_router(state: State):
    msg = router.invoke([
        SystemMessage(content="You are a router that decides whether to create a story, poem, or joke based on user input."),
        HumanMessage(content=f"Decide whether to create a 'story', 'poem', or 'joke' based on the following input: '{state['input']}'. "
                     "Respond with only one of the following options: story, poem, joke.")]
    )
    decision = msg.step.lower()
    if decision not in ["story", "poem", "joke"]:
        decision = "story"  # Default to story if unclear
    return {"decision": decision}

def output_writer(state: State):
    safe_title = re.sub(r'[\\/*?:"<>|\n]', "", state["input"]).strip()
    if len(safe_title) > 20:
        safe_title = safe_title[:20]
    base_name = safe_title
    counter = 0
    while True:
        filename = f"{base_name}.txt" if counter == 0 else f"{base_name}_{counter}.txt"
        if not os.path.exists(filename):
            break
        counter += 1
    with open(filename, "w", encoding="utf-8") as f:
        f.write(state["input"] + "\n\n"+"-"*100+"\n\n")
        f.write(state["output"])

    print(f"Content saved as {filename}")

def route_checker(state: State):
    if state["decision"] == "story":
        return "story_generator"
    elif state["decision"] == "poem":
        return "poem_generator"
    elif state["decision"] == "joke":
        return "joke_generator"
    else:
        return "story_generator"  

graph = StateGraph(State)
graph.add_node("decide_router", decide_router)
graph.add_node("story_generator", story_generator)
graph.add_node("poem_generator", poem_generator)
graph.add_node("joke_generator", joke_generator)
graph.add_node("output_writer", output_writer)
graph.add_edge(START, "decide_router")
graph.add_conditional_edges("decide_router",route_checker,{"story_generator": "story_generator",
                                                          "poem_generator": "poem_generator", "joke_generator": "joke_generator"})
graph.add_edge("story_generator", "output_writer")
graph.add_edge("poem_generator", "output_writer")
graph.add_edge("joke_generator", "output_writer")
graph.add_edge("output_writer", END)

app = graph.compile()
app.invoke({"input": "poem about mom"})




