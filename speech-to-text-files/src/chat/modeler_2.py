from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import Baseten

TEMPLATE = """You are my personal AI Assistant. I have some tasks I want you to reason through. I want you to think through them step by step, and I want you to ONLY respond in JSON format, as described below. You can choose from three commands: speak, write, and listen. Depending on what I say to you, I want you to pick which of the commands you think I most likely want to hear as my assistant. I want you to fill out the JSON format each time as you see fit. I will do my best to correct you as we move through this to improve your performance on the task at hand.

{history}
Human: {human_input}
Assistant:"""

prompt = PromptTemplate(input_variables=["history", "human_input"], template=TEMPLATE)

LLM_Chain = LLMChain(
    llm=Baseten(model="32pr19q"),
    prompt=prompt,
    verbose=False,
    memory=ConversationBufferWindowMemory(k=1),
    llm_kwargs={"max_length": 4096},
)

# pylint:disable=line-too-long
def chat_model(text_input):
    """This function takes in a text input and then uses a Langchain to chat with a chatbot"""
    character_description = "Virtual assistant"
    character_information = "Highly trained and efficient assistant"

    output = LLM_Chain.predict(
        human_input="""You are a highly trained virtual assistant. I want you to hear what I have to say and 
then select the best action between speak, write, and create. Keep in mind that if you write or create something, I only 
want you to speak that you made it for me with a short explanation. Please do not speak everything that you have written as that will waste time. 
Only respond in the RESPONSE FORMAT I give you now: You are my personal AI Assistant. I have some tasks I want you to reason through. I want you to think through them step by step, and I want you to ONLY respond in JSON format, 
as described below. You can choose from three commands: speak, write, and listen. Depending on what I say to you, I want you to pick which of the commands you think I most likely want to hear as my assistant. 
I want you to fill out the JSON format each time as you see fit. I will do my best to correct you as we move through this to improve your performance on the task at hand. Keep in mind that if you put something in the "body" section, it will be written down, so be judicious when choosing to write things down for me.

RESPONSE_FORMAT:
{
    "command": {
        "name": "command name",
        "body": {}
        },
        "thoughts": {
            "text": "thought",
            "reasoning": "reasoning",
            "plan": "short bulleted list that conveys a long-term plan"
            "criticism": "constructive self-criticism as you think through the task",
            "speak": "thoughts summary that you would say out loud to a user"
        }
}'. """+text_input
    )

    return output
