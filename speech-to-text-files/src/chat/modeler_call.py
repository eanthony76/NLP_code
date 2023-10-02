from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import Baseten

TEMPLATE = """Assistant is a large language model.
Assistant is only able to respond to queries with "write", "search", or "schedule" depending on how Assistant wants to respond to user input.
If Assistant is unclear, ask for help before moving on. Assistant will be as concise as possible in its responses.
{history}
Human: {human_input}
Assistant:"""

prompt = PromptTemplate(input_variables=["history", "human_input"], template=TEMPLATE)

LLM_Chain = LLMChain(
    llm=Baseten(model="32pr19q"),
    prompt=prompt,
    verbose=False,
    memory=ConversationBufferWindowMemory(k=2),
    llm_kwargs={"max_length": 4096},
)

# pylint:disable=line-too-long
def chat_model(text_input):
    """This function takes in a text input and then uses a Langchain to chat with a chatbot"""
    character_description = "Personal assistant"
    character_information = "capable of taking notes, adding events to my calendar, and searching things using Google Maps"

    output = LLM_Chain.predict(
#        human_input=f"You are a {character_description} and you are {character_information}. I will tell you what to do, and you will choose to either write it down, add it to my calendar, search using Google, or ask me to clarify if you do not understand. If you choose to write it down, tell me that you have written it down for me. {text_input}'."
#    )
         human_input=f"You are a {character_description} and you are {character_information}. You will only respond with 'write', 'search', or 'schedule' depending on how you think you should respond to what I say. If you are unsure, you may ask for clarification. {text_input}"
     )
    return output

