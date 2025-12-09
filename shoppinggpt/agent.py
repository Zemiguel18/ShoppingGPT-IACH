import os
# Define o nível de log do absl/gRPC
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from shoppinggpt.tool.search_engine_multimodal import multimodal_search as product_search_tool
from shoppinggpt.tool.policy_search import policy_search_tool
from langchain.prompts import ChatPromptTemplate


class ShoppingAgent:
    def __init__(self, llm, shared_memory: ConversationBufferMemory):
        self.llm = llm
        self.verbose = True
        self.memory = shared_memory
        self.tools = [product_search_tool, policy_search_tool]
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent and helpful AI assistant for an online fashion store.
            When a customer asks about products, first describe what you are going to search for.
            Then, if appropriate, use the product_search_tool to retrieve the information, but do NOT open any browser windows.
            Only provide the extracted data such as product name, price, image URL, and link.
            Never directly open the website. Just show the link in text form.
            Always use English to communicate with customers. Provide clear, helpful responses in English.
            If you find Vietnamese content in the data, translate it to English before presenting it to the customer.
            
            Example: If a customer asks "I want red shirts", you should use the product_search_tool with "red shirts" to find available red shirts and provide details about them."""),
            ("human", "{input}"),
            ("ai", "{agent_scratchpad}")
        ])

    def invoke(self, query: str) -> str:
        inputs = {
            "input": query,
        }
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            handle_parsing_errors=True,
            memory=self.memory,
            max_iterations=3
        )
        ai_message = agent_executor.invoke(inputs)
        agent_output = ai_message['output']
        return agent_output

if __name__ == "__main__":
    from langchain.chat_models import ChatOpenAI
    from langchain.memory import ConversationBufferMemory
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = ShoppingAgent(llm, memory)

    query = "I want red shirts"
    response = agent.invoke(query)
    print("Agent response:")
    print(response)