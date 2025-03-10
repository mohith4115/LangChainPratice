from langchain_openai import AzureChatOpenAI
from langchain.chains.sequential import SimpleSequentialChain,SequentialChain
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
    model=os.getenv("MODEL_NAME"),
    api_version=os.getenv("api_version"),
    max_tokens=4096
)

product = "a device to play racing games"

template1 = "give a good name for this company that makes this {product}"

template2 = "give me the description of the company {input} in 20 words"

prompt1 = ChatPromptTemplate.from_template(template1)

prompt2 = ChatPromptTemplate.from_template(template2)

chain1 = prompt1 | llm
chain2 = prompt2 | llm

#deprecated 
# overall_chain = SimpleSequentialChain(chains=[chain1,chain2],
#                                       verbose=True)

overall_chain = chain1 | chain2
res = overall_chain.invoke({"product":product})
print(res.content)