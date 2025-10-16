#-------------------------------------------------------------------------------------------------------------------------------------
#|  THIS FILE DEALS WITH BUILDING AN AGENT OVER ReAct pattern                                                        |
# -------------------------------------------------------------------------------------------------------------------------------------
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_tavily import TavilySearch
# from langchain_tavily import TavilySearch

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2,doc_content_chars_max=500)
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=2,doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
tavily = TavilySearch()
memory = MemorySaver()
# few custom tools
def add(a: int, b: int) -> int:
    """
    Add two integers together.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The sum of a and b.
    """
    return a + b


def multiply(a: int, b: int) -> int:
    """
    Multiply two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The product of a and b.
    """
    return a * b


def divide(a: int, b: int) -> int:
    """
    Divide one integer by another using integer division.

    Args:
        a (int): The numerator.
        b (int): The denominator (must not be zero).

    Returns:
        int: The result of integer division (a // b).

    Raises:
        ZeroDivisionError: If b is zero.
    """
    return a // b



# arxiv_result = arxiv.invoke("Attention is all you need")
# print(arxiv_result)

# result = tavily.invoke({"query": "Latest research on AI in healthcare","num_results":2})
# print(result)

# result = wiki.invoke("Artificial Intelligence")
# print(result)

tools = [arxiv, wiki, tavily, add, multiply, divide]
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.4)
llm_with_tools = llm.bind_tools(tools)
# ai_result = llm_with_tools.invoke([HumanMessage(content="who is mark zuckerberg?")])
# print(ai_result.tool_calls)

# Node definition

class State(TypedDict):
    messages: Annotated[list, add_messages]

def tool_calling_llm(state: State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm",tools_condition)
builder.add_edge("tools", "tool_calling_llm")
graph = builder.compile(checkpointer=memory)
# messages = graph.invoke(
#     {"messages": [HumanMessage(content="what is my name?")]},
#     config={"configurable": {"thread_id": "1"}}
# )
# for m in messages["messages"]:
#     m.pretty_print()

# STREAMING IN THE GRAPH

while True:
    config = {"configurable":{"thread_id":"1"}}
    user_query = input("User: ")
    if user_query.lower() in ["exit","quit"]:
        break
    for chunk in graph.stream( {"messages": [HumanMessage(content=user_query)]}, config={"configurable": {"thread_id": "1"}}, stream="values"):
        print(chunk)
    
