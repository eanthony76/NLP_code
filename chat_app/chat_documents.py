from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from embed_documents import embed_langchain
from langchain.memory import ConversationSummaryMemory

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

question = "What is the fox's name?"

vectorstore = embed_langchain(["The dog's name is Auggie", "The fox's name is Ted", "It was the first time I have ever seen it!", "The quick brown fox jumped over the lazy dog"], 'ms-marco-MiniLM-L-12-v2')

llm = HuggingFacePipeline.from_model_id(
    model_id='google/flan-t5-base',
    task='text2text-generation',
    model_kwargs={'temperature': .8, "max_length": 64},
)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={'prompt': QA_CHAIN_PROMPT}
)
result = qa_chain({"query": question})
print(result['result'])