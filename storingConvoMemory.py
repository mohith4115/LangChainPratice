from langchain_openai import AzureChatOpenAI
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

template = """
    this is the "conversation history : {history}
    you are a sr software developer give me code for {input}
"""

prompt = PromptTemplate(template=template,input_variables=["history","input"])

load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
    model=os.getenv("MODEL_NAME"),
    api_version=os.getenv("api_version"),
    max_tokens=4096
)
memory = ConversationBufferMemory()

chain = ConversationChain(llm=llm,memory=memory,prompt=prompt,verbose=True)


res1 = chain.invoke(input="write python program for finding a prime number")
print(res1)
res2 = chain.invoke(input="write python program for finding a amstrong number")
print(res2)

res3 = chain.invoke(input="write a c program for adding 2 numbers and give me the programing language name i asked you to write prime number code in")
print(res3)

print(memory.buffer)

