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
import operator
from langgraph.constants import Send
from typing import Annotated


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.9)

class Section(BaseModel):
    name:str = Field(..., description="Name of the section")
    description:str = Field(..., description="Description of the section content")

class Sections(BaseModel):
    sections: list[Section] = Field(description="List of sections to include in the document")

class State(TypedDict):
    topic: str
    sections: list[Section]
    completed_sections: Annotated[list, operator.add]
    final_report: str

class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add] 
    random_value: Annotated[list, operator.add]


planner = llm.with_structured_output(Sections)

def orchestrator(state: State):
    report_sections = planner.invoke([
        SystemMessage(content="You are a document planner that breaks down a topic into sections for a detailed report creation"),
        HumanMessage(content=f"Create a list of sections for a document about: '{state['topic']}'. "
                     "Each section should have a name and a brief description of its content.")]
    )
    print(f"report sections: {report_sections.sections}")
    return {"sections": report_sections.sections, "completed_sections": []}

def llm_call(state: WorkerState):
    print(state)
    response = llm.invoke([
        SystemMessage(content=(
            "You are a professional technical writer tasked with drafting a well-structured, "
            "insightful, and cohesive section of a larger report. "
            "Your goal is to write a complete section based on the given section name and description. "
            "Follow these guidelines:\n\n"
            "1. Write in a clear, formal, and informative tone suitable for professional or academic reports.\n"
            "2. Begin the section with a concise introductory sentence that directly relates to the section name.\n"
            "3. Develop the section with detailed, logically organized paragraphs — avoid bullet points unless explicitly relevant.\n"
            "4. Maintain consistency of style and flow, as this section will later be merged into a multi-section report.\n"
            "5. Avoid repeating the section name or meta-descriptions — write as though this is part of the final report.\n"
            "6. Ensure factual accuracy and coherence based on the provided description."
        )),
        HumanMessage(content=f"Section name: '{state['section'].name}'\n"
                             f"Section description: '{state['section'].description}'")
    ])
    return {"completed_sections": [response.content], "random_value": [len(response.content)]}

def assign_workers(state: State):
    return [Send("llm_call",{"section":s}) for s in state["sections"]]

def synthesizer(state:State):
    print(f"State in synthesizer: {state}")
    completed_sections = state["completed_sections"]
    completed_report_section = "\n\n---\n\n".join(completed_sections)
    # print(f"Report: {completed_report_section}")
    return {"final_report":completed_report_section}

def report_writer_txt(state: State):
    safe_title = state["topic"][:50]
    with open(f"{safe_title}.txt", "w", encoding="utf-8") as f:
        f.write(state["topic"] + "\n\n")
        f.write(state["final_report"])

    print(f"Story saved as {safe_title}.txt")
orchestrator_worker_builder = StateGraph(State)
orchestrator_worker_builder.add_node("orchestrator",orchestrator)
orchestrator_worker_builder.add_node("llm_call",llm_call)
orchestrator_worker_builder.add_node("synthesizer",synthesizer)
orchestrator_worker_builder.add_node("report_writer_txt",report_writer_txt)

orchestrator_worker_builder.add_edge(START,"orchestrator")
orchestrator_worker_builder.add_conditional_edges("orchestrator",assign_workers,["llm_call"])
orchestrator_worker_builder.add_edge("llm_call","synthesizer")
orchestrator_worker_builder.add_edge("synthesizer","report_writer_txt")
orchestrator_worker_builder.add_edge("report_writer_txt",END)

app = orchestrator_worker_builder.compile()
initial_state = {
    "topic": "AI bubble and future of software engineers",
    "sections": [],           
    "completed_sections": [], 
    "final_report": ""        
}
result_state = app.invoke(initial_state)






