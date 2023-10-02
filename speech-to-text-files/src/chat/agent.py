# coding: utf-8
get_ipython().run_line_magic('pip', 'install openai google-search-results')
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
# Initialize the OpenAI language model
# Replace <your_api_key> in openai_api_key="<your_api_key>" with your actual OpenAI key.
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

# Initialize the SerpAPIWrapper for search functionality
# Replace <your_api_key> in serpapi_api_key="<your_api_key>" with your actual SerpAPI key.
search = SerpAPIWrapper(serpapi_api_key='ebc24973be712e5e2ad03c7ca0a3f5b141626c1aadd1a42f6f34e0b7ecdf8f1b')
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful when you need to answer questions about current events. You should ask targeted questions.",
    ),
]
mrkl = initialize_agent(
    tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True
)
# Do this so we can see exactly what's going on under the hood
import langchain

langchain.debug = True
mrkl.run("What is the weather in LA and SF?")
