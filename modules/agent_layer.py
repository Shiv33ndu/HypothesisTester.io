# for using local LLM models
# from langchain_ollama.llms import OllamaLLM    

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from dotenv import load_dotenv
import os
from modules.agent_instruct import classify, reading
from utils.data_reader import prepare_data_context
import json
import re

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
reading_prompt = ChatPromptTemplate.from_template(reading())
reading_chain = reading_prompt | gemini_llm
# ============================================================


def handle(user_prompt : str, df : pd.DataFrame, columns : list[str]):
    print(user_prompt)
    print(df.head()) 

    data_context_json = prepare_data_context(df, columns)

    # we will pass this JSON to simple model to know if it can read every element of this 
    for chunk in reading_chain.stream({'data_context_json': data_context_json}):
        print(chunk.content, end='', flush=True)


    """

    # for chunk in chain.stream({"user_prompt": user_prompt, "dataframe": df}):
    #     print(chunk, end='', flush=True)
    res = ''
    for chunk in classify_chain.stream({"user_prompt": user_prompt, "dataframe": df, "columns":columns}):
        print(chunk.text, end='', flush=True)
        res += chunk.text
    
    # Use a regex to extract the content between the first { and the last }
    json_match = re.search(r'\{.*\}', res, re.DOTALL)
    
    if json_match:
        # If a JSON-like object is found, use that for parsing
        json_string = json_match.group(0)
    else:
        # Fallback to the raw string if no JSON is found
        json_string = res
        
    try:
        response = json.loads(json_string)
        return response
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from LLM: {e}")
        print("Raw LLM response:")
        print(repr(res)) # Using repr() shows hidden characters like newlines
        return None
    """