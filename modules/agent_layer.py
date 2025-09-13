# for using local LLM models
# from langchain_ollama.llms import OllamaLLM    

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from modules.agent_instruct import classify, classify_and_structure, guard_prompt
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

# classification model (might remove later one)
classify_prompt = ChatPromptTemplate.from_template(classify())
classify_chain = classify_prompt | gemini_llm


# classify and structure model
reading_prompt = ChatPromptTemplate.from_template(classify_and_structure())
reading_chain = reading_prompt | gemini_llm


# gatekeep and check the user question model
check_prompt = ChatPromptTemplate.from_template(guard_prompt())
check_chain = check_prompt | gemini_llm

# ============================================================


CANONICAL = {
"One-Sample t-test",
"Two-Sample Independent t-test",
"Paired t-test",
"One-Sample Z-test",
"ANOVA",
"Kruskal-Wallis H Test",
"Chi-Square Test of Independence",
"Chi-Square Goodness-of-Fit Test",
"Fisher's Exact Test",
"Correlation Test (Pearson/Spearman)",
"Linear Regression Analysis",
"Mann-Whitney U Test",
"Wilcoxon Signed-Rank Test",
}


# handles all the main task if user_question is inferential hypothesis  
def handle(user_prompt : str, df : pd.DataFrame, columns : list[str]):
    # print(user_prompt)
    # print(df.head()) 

    data_context_json = prepare_data_context(df, columns)  # using data_reader to get the whole context of the data for the LLM
    
    res = ''

    # we will pass this JSON to simple model to infer from this and give us structured variables by classifying the test and parameters 
    for chunk in reading_chain.stream({'data_context_json': data_context_json, 'dataframe': df, 'user_prompt': user_prompt}):
        # print(chunk.content, end='', flush=True)
        res += chunk.content
    
    llm_response = give_json(res)                     # we converting the received json like string to actual json format

    print(f"\n\nLLM RESPONSE(agent_layer)----\n {llm_response}")

    if llm_response.get("test_name") not in CANONICAL:
        return {'Error:' : f"LLM Response doesn't have the correct Test Name : {llm_response.get("test_name")}"}

    else:
        response, test_results = dispatch_test(llm_response, df)     # now we got the test names, and columns, test params, we need to perform Hypothesis Testing on this
        
        print(f"\n\n------- RESPONSE & TEST RESULT -------\n{response}\n{test_results}")

        return response, test_results
    

# checks if the user's question is inferential or descriptive(we wont need hypothesis for that)
def check_user_question(user_prompt: str, df: pd.DataFrame):
    res = ''

    for chunks in check_chain.stream({'user_prompt' : user_prompt, 'data_context': df.head().to_string()}):
        res += chunks.content


    checked_question_json = give_json(res)

    return checked_question_json  