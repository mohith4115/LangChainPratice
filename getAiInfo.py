from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

template = """
    Explain and give the latest information about this ai tool/framework/concept/methodlogy :{tool}
    And what is the use of it and how to use it
"""

prompt = PromptTemplate(input_variables=["tool"],template=template)

load_dotenv()

llm = ChatOpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("OPEN_API_KEY"),
    temperature=0,
    model=os.getenv("MODEL_NAME"),
    max_tokens=4096
)

chain = prompt | llm

chain.invoke(input = {"tool":"langfuse"})