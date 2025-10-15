#-------------------------------------------------------------------------------------------------------------------------------------
#|  THE MAIN AIM OF THIS FILE IS TO DEMONSTRATE HOW TO CREATE A CHATBOT THAT CAN USE MULTIPLE TOOLS BASED ON USER INPUT.              |
#| THE BOT WILL DECIDE WHICH TOOL TO USE BASED ON THE USER'S REQUEST.                                                                 |
#|  TOOLS WE'LL BE USING ARE ARXIV, WIKIPIDEA SERACH AND SOME CUSTOM TOOLS.                                                           |
# -------------------------------------------------------------------------------------------------------------------------------------
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
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

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2,doc_content_chars_max=500)
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=2,doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
tavily = TavilySearch()

# arxiv_result = arxiv.invoke("Attention is all you need")
# print(arxiv_result)

# result = tavily.invoke({"query": "Latest research on AI in healthcare","num_results":2})
# print(result)

# result = wiki.invoke("Artificial Intelligence")
# print(result)

tools = [arxiv, wiki, tavily]
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
builder.add_edge("tools", END)
graph = builder.compile()
messages = graph.invoke({"messages":[HumanMessage(content="provide me top 10 recent AI news")]} )
for m in messages["messages"]:
    m.pretty_print()


