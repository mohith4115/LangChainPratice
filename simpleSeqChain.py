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

template2 = "give me the description of the company in 20 words"

prompt1 = ChatPromptTemplate(template_format=template1)

prompt2 = ChatPromptTemplate(template_format=template2)

chain1 = prompt1 | llm
chain2 = prompt2 | llm

overall_chain = SimpleSequentialChain(chains=[chain1,chain2],
                                      verbose=True)

overall_chain.run(product)