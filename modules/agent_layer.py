# for using local LLM models
# from langchain_ollama.llms import OllamaLLM    

from google import genai
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from dotenv import load_dotenv
import os
from modules.agent_instruct import classify
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

gemini = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
model = gemini.models.generate_content_stream 
template = classify()
prompt = ChatPromptTemplate.from_template(template)

# ============================================================



def handle(user_prompt : str, df : pd.DataFrame, columns : list[str]):
    print(user_prompt)
    print(df.head()) 

    # for chunk in chain.stream({"user_prompt": user_prompt, "dataframe": df}):
    #     print(chunk, end='', flush=True)
    res = ''
    for chunk in model(model='gemini-1.5-flash', 
                       contents=template.format(user_prompt=user_prompt, dataframe=df, columns=columns)):
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