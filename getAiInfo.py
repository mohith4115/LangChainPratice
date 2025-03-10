from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

template = """
    Explain and give the latest information about this ai tool/framework/concept/methodlogy :{tool}
    And what is the use of it and how to use it and give an example related to this {example}
"""

prompt = PromptTemplate(input_variables=["tool","example"],template=template)

load_dotenv()

# llm = ChatOpenAI(
#     base_url=os.getenv("BASE_URL"),
#     api_key=os.getenv("OPEN_API_KEY"),
#     temperature=0,
#     model=os.getenv("MODEL_NAME"),
#     max_tokens=4096
# )

#when connecting to a deployed model in azure
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
    model=os.getenv("MODEL_NAME"),
    api_version=os.getenv("api_version"),
    max_tokens=4096
)

chain = prompt | llm

res = chain.invoke(input = {"tool":"langfuse","example":"in context of refining an input jaava code"})
print(res)