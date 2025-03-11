from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
    model=os.getenv("MODEL_NAME"),
    api_version=os.getenv("api_version"),
    max_tokens=4096
)


animal_facts_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You like telling facts and you tell facts about {animal}."),
        ("human", "Tell me {count} facts."),
    ]
)

#prompt template for translation to French
translation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a translator and convert the provided text into {language}."),
        ("human", "Translate the following text to {language}: {text}"),
    ]
)

# Define additional processing steps using RunnableLambda
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")
# prepare_for_translation = RunnableLambda(lambda output: {"text": output, "language": "french"})

# we can use a normal function as well upto us
def prepare_for_translation(output):
    return {"text": output, "language": "french"}


# Create the combined chain using LangChain Expression Language
chain = animal_facts_template | llm | StrOutputParser() | prepare_for_translation | translation_template | llm | StrOutputParser() 

# Run the chain
result = chain.invoke({"animal": "cat", "count": 2})

# Output
print(result)