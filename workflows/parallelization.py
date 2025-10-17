from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from fpdf import FPDF
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
import re

# Load environment
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.4)

class State(TypedDict):
    topic: str
    charachters: str
    story_premise: str
    final_Story: str
    story_setting: str
    story_title: str

def title_generator(state: State):
    msg = llm.invoke(f"Generate a catchy title for a story on the topic '{state['topic']}' story premise '{state['story_premise']}' and story setting '{state['story_setting']}' Return only the title, nothing else.")
    return {"story_title": msg.content}

def generate_story_premise(state: State):
    msg = llm.invoke(f"Write one story premise on the topic '{state['topic']}''.")
    return {"story_premise": msg.content}

def generate_story_setting(state: State):
    msg = llm.invoke(f"Generate an interesting setting for the story topic: '{state['topic']}'")
    return {"story_setting": msg.content}

def generate_charachters(state: State):
    msg = llm.invoke(f"Create main charachters for the story topic: '{state['topic']}'")
    return {"charachters": msg.content}

def finalize_story(state: State):
    msg = llm.invoke(
        f"Using the story premise: '{state['story_premise']}', setting: '{state['story_setting']}', "
        f"and charachters: '{state['charachters']}', write a complete engaging story."
    )
    return {"final_Story": msg.content}

# ---- NEW MERGE NODE ----
def merge_story_elements(state: State):
    """Combine story premise, setting, and characters into a single state dict."""
    return {
        "story_premise": state.get("story_premise", ""),
        "story_setting": state.get("story_setting", ""),
        "characters": state.get("characters", "")
    }

def story_writer_txt(state: State):
    # Sanitize title for filename
    safe_title = re.sub(r'[\\/*?:"<>|\n]', "", state["story_title"]).strip()
    if len(safe_title) > 100:
        safe_title = safe_title[:100]

    # Open a text file for writing
    with open(f"{safe_title}.txt", "w", encoding="utf-8") as f:
        # Write title
        f.write(state["story_title"] + "\n\n")
        # Write the story content
        f.write(state["final_Story"])

    print(f"Story saved as {safe_title}.txt")

graph = StateGraph(State)
graph.add_node("generate_story_premise", generate_story_premise)
graph.add_node("generate_story_setting", generate_story_setting)
graph.add_node("generate_charachters", generate_charachters)
graph.add_node("merge_story_elements", merge_story_elements)
graph.add_node("title_generator", title_generator)
graph.add_node("finalize_story", finalize_story)
graph.add_node("story_writer_txt", story_writer_txt)

graph.add_edge(START, "generate_story_premise")
graph.add_edge(START, "generate_story_setting")
graph.add_edge(START, "generate_charachters")
graph.add_edge("generate_story_premise", "merge_story_elements")
graph.add_edge("generate_story_setting", "merge_story_elements")
graph.add_edge("generate_charachters", "merge_story_elements")
graph.add_edge("merge_story_elements", "finalize_story")
graph.add_edge("finalize_story", "title_generator")
graph.add_edge("title_generator", "story_writer_txt")
graph.add_edge("story_writer_txt", END)

app = graph.compile()
app.invoke({"topic":"A comedy story about guest arrival"})
