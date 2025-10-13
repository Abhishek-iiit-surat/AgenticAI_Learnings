from dotenv import load_dotenv
import os
import pprint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name="gpt-5-nano",temperature=0.4)
messages = [HumanMessage(content=f"Hello, who are you?")]
messages.append(AIMessage(content=f"I am an AI created by OpenAI. How can I help you today?"))
messages.append(SystemMessage(content=f"Keep the responses concise and to the point."))
messages.append(HumanMessage(content=f"what is pydantic in python? "))


for message in messages:
    message.pretty_print()  

result = llm.invoke(messages)
print(result.content)
