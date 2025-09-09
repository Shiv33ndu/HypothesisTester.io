# for using local LLM models
# from langchain_ollama.llms import OllamaLLM    

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from modules.agent_instruct import classify, reading, classify_and_structure
from utils.data_reader import prepare_data_context
from utils.jsonfy import give_json
from modules.hypothesis_test import dispatch_test



load_dotenv()

# =============== Code to use Local LLMs ====================

# model = OllamaLLM(model='phi3:3.8b')
# template = classify()
# prompt = ChatPromptTemplate.from_template(template)
# chain = prompt | model

# ============================================================


# ==================== API Called LLMs =======================

gemini_llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash", 
                                    google_api_key=os.getenv('GEMINI_API_KEY'),
                                    )

# ============================================================

# ========================= Chain & Prompts ==================

# classification model
classify_prompt = ChatPromptTemplate.from_template(classify())
classify_chain = classify_prompt | gemini_llm


# reading model
reading_prompt = ChatPromptTemplate.from_template(classify_and_structure())
reading_chain = reading_prompt | gemini_llm
# ============================================================


def handle(user_prompt : str, df : pd.DataFrame, columns : list[str]):
    # print(user_prompt)
    # print(df.head()) 

    data_context_json = prepare_data_context(df, columns)  # using data_reader to get the whole context of the data for the LLM
    
    res = ''

    # we will pass this JSON to simple model to infer from this and give us structured variables by classifying the test and parameters 
    for chunk in reading_chain.stream({'data_context_json': data_context_json, 'dataframe': df, 'user_prompt': user_prompt}):
        # print(chunk.content, end='', flush=True)
        res += chunk.content
    
    response = give_json(res)                     # we converting the received json like string to actual json format


    response, test_results = dispatch_test(response, df)     # now we got the test names, and columns, test params, we need to perform Hypothesis Testing on this

    return response, test_results
    
