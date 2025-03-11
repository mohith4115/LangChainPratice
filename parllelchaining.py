from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda,RunnableParallel
from langchain_openai import AzureChatOpenAI
import os

load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
    model=os.getenv("MODEL_NAME"),
    api_version=os.getenv("api_version"),
    max_tokens=4096
)

curry = ChatPromptTemplate.from_messages(
    [
        ("system","you are a head cook for a big family"),
        ("human","suggest any one dish that contains this item :{item}")
    ]
)

ingredients = ChatPromptTemplate.from_messages(
    [
        ("system","you are a cooking maid for a big family"),
        ("human","give a note of all the ingredients for this dish:{dish}")
    ]
)

cooking_steps = ChatPromptTemplate.from_messages(
    [
        ("system","you are a senior cook for a big family"),
        ("human","give step by step process of making this dish with given ingredients:{ingredients}")
    ]
)

seasoning = ChatPromptTemplate.from_messages(
    [
        ("system","you are a head cook for a big family"),
        ("human","suggest the seasoning on this dish with following procedure and only give seasoning in output:{process}")
    ]
)

starter = ChatPromptTemplate.from_messages(
    [
        ("system","you are a head cook for a big family"),
        ("human","suggest the starter that goes well with this process:{process}")
    ]
)

curryOutput = RunnableLambda(lambda dish:  {"dish":dish})
ingredientsOutput = RunnableLambda(lambda ingredients:  {"ingredients":ingredients})
processOutput = RunnableLambda(lambda process:  {"process":process})
seasoningPrompt = RunnableLambda(lambda x: seasoning.format_prompt(process=x["process"]))
starterPrompt = RunnableLambda(lambda x: starter.format_prompt(process=x["process"]))

def debug_print(stage):
    return RunnableLambda(lambda x: print(f"\n=== {stage} Output ===\n{x}") or x)

chain = (
    curry |
    llm |
    StrOutputParser() |
    debug_print("Dish Suggested") |  # 👈 Print after LLM generates the dish
    curryOutput |
    ingredients |
    llm |
    StrOutputParser() |
    debug_print("Ingredients List") |  # 👈 Print after LLM generates ingredients
    ingredientsOutput |
    cooking_steps |
    llm |
    StrOutputParser() |
    debug_print("Cooking Steps") |  # 👈 Print after LLM generates cooking steps
    processOutput 
)

starterChain = starterPrompt | llm | StrOutputParser() | debug_print("Starter Suggested")
seasoningChain = seasoningPrompt | llm | StrOutputParser() | debug_print("Seasoning Suggested")

parallel_chain = RunnableParallel({
    "seasoning": seasoningChain,
    "starter": starterChain
})

final_chain = chain | parallel_chain

result = final_chain.invoke({"item": "chicken"})
print("\n=== Final Output ===\n", result)

result = final_chain.invoke({"item":"chicken"})
