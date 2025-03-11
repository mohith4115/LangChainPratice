from langchain_openai import AzureChatOpenAI
from langchain.chains.sequential import SimpleSequentialChain,SequentialChain
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
import os
from langchain.schema.output_parser import StrOutputParser
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


# Load environment variables from .env
load_dotenv()


# Define prompt templates (no need for separate Runnable chains)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows facts about {animal}."),
        ("human", "Tell me {fact_count} facts."),
    ]
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | llm | StrOutputParser()
# chain = prompt_template | model

# Run the chain
result = chain.invoke({"animal": "elephant", "fact_count": 1})

# Output
print(result)