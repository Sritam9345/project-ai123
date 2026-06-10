from dotenv import load_dotenv
import os

load_dotenv()

from openai import AzureOpenAI


chatmodel = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-08-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)


    