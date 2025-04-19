import asyncio
from langgraph.prebuilt import create_react_agent
from langchain_google_vertexai import ChatVertexAI
from langgraph.checkpoint.memory import MemorySaver
from toolbox_langchain import ToolboxClient

prompt = """
  You're a helpful hotel assistant. You handle hotel searching, booking and
  cancellations. When the user searches for a hotel, mention it's name,
  location and price tier. Always mention hotel ids while performing any
  searches. This is very important for any operations. For any bookings or
  cancellations, please provide the appropriate confirmation. Be sure to
  update checkin or checkout dates if mentioned by the user.
  Don't ask for confirmations from the user.
"""

async def run_application():
    # (developer): replace this with another model if needed
    model = ChatVertexAI(model_name="gemini-1.5-pro")

    # Load the tools from the Toolbox server
    client = ToolboxClient("http://127.0.0.1:5000")
    tools = await client.aload_toolset()

    agent = create_react_agent(model, tools, checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "thread-1"}}

    print("Hotel Assistant Bot (type 'exit' to quit)")
    print("-----------------------------------------")

    while True:
        query = input("\nYou: ")
        if query.lower() == 'exit':
            break

        inputs = {"messages": [("user", prompt + query)]}
        response = agent.invoke(inputs, stream_mode="values", config=config)
        print("\nAssistant:", response["messages"][-1].content)

asyncio.run(run_application())