from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain_openai import AzureChatOpenAI
import os

load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
    model=os.getenv("MODEL_NAME"),
    api_version=os.getenv("api_version"),
    max_tokens=4096,
)

curry = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a head cook for a big family"),
        ("human", "suggest any one dish that contains this item :{item}"),
    ]
)

ingredients = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a cooking maid for a big family"),
        ("human", "give a note of all the ingredients for this dish:{dish}"),
    ]
)

cooking_steps = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a senior cook for a big family"),
        ("human", "give step by step process of making this dish with given ingredients:{ingredients}"),
    ]
)

seasoning = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a head cook for a big family"),
        ("human", "suggest the seasoning on this dish with following procedure and only give seasoning in output:{process}"),
    ]
)

starter = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a head cook for a big family"),
        ("human", "suggest the starter that goes well with this process:{process}"),
    ]
)


#these runnable lambdas are required as next prompt template excepts dict 

curryOutput = RunnableLambda(lambda dish: {"dish": dish})
ingredientsOutput = RunnableLambda(lambda ingredients: {"ingredients": ingredients})
processOutput = RunnableLambda(lambda process: {"process": process})
seasoningPrompt = RunnableLambda(lambda x: seasoning.format_prompt(process=x["process"]))
starterPrompt = RunnableLambda(lambda x: starter.format_prompt(process=x["process"]))

def debug_print(stage):
    return RunnableLambda(lambda x: print(f"\n=== {stage} Output ===\n{x}") or x)

def print_formatted_prompt(stage, prompt_template):
    def inner(input_data):
        formatted_prompt = prompt_template.format_prompt(**input_data)
        print(f"\n=== {stage} Formatted Prompt ===\n{formatted_prompt.to_string()}")
        return input_data
    return RunnableLambda(inner)

chain = (
    RunnableLambda(lambda x: x) | print_formatted_prompt("Curry Prompt", curry) | curry |
    llm |
    StrOutputParser() |
    debug_print("Dish Suggested") |
    curryOutput |
    print_formatted_prompt("Ingredients Prompt", ingredients) | ingredients |
    llm |
    StrOutputParser() |
    debug_print("Ingredients List") |
    ingredientsOutput |
    print_formatted_prompt("Cooking Steps Prompt", cooking_steps) | cooking_steps |
    llm |
    StrOutputParser() |
    debug_print("Cooking Steps") |
    processOutput
)

starterChain = print_formatted_prompt("Starter Prompt", starter) | starterPrompt | llm | StrOutputParser() | debug_print("Starter Suggested")
seasoningChain = print_formatted_prompt("Seasoning Prompt", seasoning) | seasoningPrompt | llm | StrOutputParser() | debug_print("Seasoning Suggested")

parallel_chain = RunnableParallel({"seasoning": seasoningChain, "starter": starterChain})

summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a food critic summarizing a meal."),
        ("human", "Summarize the meal, including the seasoning and starter suggestions. Seasoning: {seasoning}, Starter: {starter}"),
    ]
)

summary_chain =  summary_prompt | llm |StrOutputParser()
final_chain = chain | parallel_chain | summary_chain

result = final_chain.invoke({"item": "chicken"})
print("\n=== Final Output ===\n", result)

result = final_chain.invoke({"item": "chicken"})