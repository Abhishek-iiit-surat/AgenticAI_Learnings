from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
import os

# --- Setup ---
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4.1-mini")

sys_message = SystemMessage(
    content=(
        "You are an AI assistant with access to the following mathematical tools: "
        "add, multiply, subtract, divide. Use these tools to perform calculations as needed."
    )
)

# --- Define tools ---
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

def subtract(a: int, b: int) -> int:
    """Subtract one integer from another."""
    return a - b

def divide(a: int, b: int) -> float:
    """Divide one integer by another."""
    return 0 if b == 0 else a / b

tools = [add, multiply, subtract, divide]
llm_with_tools = llm.bind_tools(tools)

# --- Assistant node ---
def assistant(state: MessagesState):
    human_decision = interrupt({
        "prompt": "About to call tool for computation. Do you want to continue?",
        "latest_message": state["messages"][-1].content
    })
    if human_decision.get("proceed", False):
        response = llm_with_tools.invoke(state["messages"] + [sys_message])
        return {"messages": state["messages"] + [response]}
    else:
        return {"messages": state["messages"] + [AIMessage(content="Operation cancelled.")], "__next__": END}

# --- Build graph ---
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
    {"tools": "tools", "__end__": END},
)
builder.add_edge("tools", "assistant")

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "123"}}
initial_input = {"messages": [HumanMessage(content="What is 123 multiplied by 2 and then plus 123?")]}
next_input = initial_input
while True:
    result = graph.invoke(next_input, config=config)
    if "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"][0]
        data = interrupt_obj.value
        print("\INTERRUPT:", data["prompt"])
        print(f"User said: {data['latest_message']}")
        user_input = input("Proceed? (yes/no): ").strip().lower()
        resume_value = {"proceed": user_input == "yes"}
        next_input = Command(resume=resume_value)
        continue  

    for event in graph.stream(None, config=config, stream_mode="values"):
        if "messages" in event:
            event["messages"][-1].pretty_print()
    break