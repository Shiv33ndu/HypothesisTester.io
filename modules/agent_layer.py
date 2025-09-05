from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd

from modules.agent_instruct import classify

model = OllamaLLM(model='llama3.2')
template = classify()
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


def handle(user_prompt : str, df : pd.DataFrame):
    print(user_prompt)
    print(df.head()) 

    for chunk in chain.stream({"user_prompt": user_prompt, "dataframe": df}):
        print(chunk, end='', flush=True)