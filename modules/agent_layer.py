# for using local LLM models
# from langchain_ollama.llms import OllamaLLM    

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.exceptions import LangChainException
from google.api_core.exceptions import ResourceExhausted

from modules.agent_instruct import classify_and_structure, guard_prompt, summary, chat
from utils.data_reader import prepare_data_context
from utils.jsonfy import give_json
from modules.hypothesis_test import dispatch_test

from typing import Dict, Any

from utils.render_chat_history import render_chat


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

# gatekeep and check the user question model
check_prompt = ChatPromptTemplate.from_template(guard_prompt())
check_chain = check_prompt | gemini_llm


# classify and structure model
reading_prompt = ChatPromptTemplate.from_template(classify_and_structure())
reading_chain = reading_prompt | gemini_llm

# summarize in natural language
summary_prompt = ChatPromptTemplate.from_template(summary())
summary_chain = summary_prompt | gemini_llm


# chat with the user about the test
chat_prompt = ChatPromptTemplate.from_template(chat())
chat_chain = chat_prompt | gemini_llm

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
    try:
        for chunk in reading_chain.stream({'data_context_json': data_context_json, 'dataframe': df, 'user_prompt': user_prompt}):
        # print(chunk.content, end='', flush=True)
            res += chunk.content
    except ResourceExhausted:
        return None, {'Error:': f"You have exhausted the free Token. Please try again later!"}, None
    
    except Exception as e:
        # Fallback in case you can't catch ResourceExhausted explicitly
        if "ResourceExhausted" in str(e) or "quota" in str(e).lower():
            return None, {"Error": "You have exceeded your quota. Please check your subscription and billing details."}, None
        return None, {"Error": "An unexpected error occurred. Please try again later."}, None

    
    llm_response = give_json(res)                     # we converting the received json like string to actual json format

    print(f"\n\nLLM RESPONSE(agent_layer)----\n {data_context_json} \n {llm_response}")

    if llm_response.get("test_name") not in CANONICAL:
        return None, {'Error:' : f"LLM Response doesn't have the correct Test Name : {llm_response.get("test_name")}"}, None

    else:
        response, test_results = dispatch_test(llm_response, df)     # now we got the test names, and columns, test params, we need to perform Hypothesis Testing on this
        
        print(f"\n\n------- RESPONSE & TEST RESULT -------\n{response}\n{test_results}")

        return data_context_json, response, test_results
    

# checks if the user's question is inferential or descriptive(we wont need hypothesis for that)
def check_user_question(user_prompt: str, df: pd.DataFrame):
    res = ''

    
    try:
        for chunks in check_chain.stream({'user_prompt' : user_prompt, 'data_context': df.head().to_string()}):
            res += chunks.content

    except ResourceExhausted:
        return None, {'Error:': f"You have exhausted the free Token. Please try again later!"}, None
    
    except Exception as e:
        # Fallback in case you can't catch ResourceExhausted explicitly
        if "ResourceExhausted" in str(e) or "quota" in str(e).lower():
            return None, {"Error": "You have exceeded your quota. Please check your subscription and billing details."}, None
        return None, {"Error": "An unexpected error occurred. Please try again later."}, None


    checked_question_json = give_json(res)

    return checked_question_json  



# A summarizer layer to summarize the statistical jargons in human launguage for non techincal users 
def summarize(user_prompt: str, llm_response: Dict[str, Any], test_results: Dict[str, Any], plot_context: Dict[str, Any]):

    
    
    try:
        for chunks in summary_chain.stream({
        'user_prompt': user_prompt, 
        'test_name': llm_response.get('test_name', None), 
        'columns': llm_response.get('columns', []),
        'H0_statement': llm_response.get('hypotheses', {}).get('H0', None), 
        'H1_statement': llm_response.get('hypotheses', {}).get('H1', None),
        'reasoning': llm_response.get('reasoning', None),
        'test_results': test_results,
        'plot_context': plot_context}):
            print(chunks.content, end='', flush=True)
            yield chunks.content
            
    except ResourceExhausted:
        return None, {'Error:': f"You have exhausted the free Token. Please try again later!"}, None
    
    except Exception as e:
        # Fallback in case you can't catch ResourceExhausted explicitly
        if "ResourceExhausted" in str(e) or "quota" in str(e).lower():
            return None, {"Error": "You have exceeded your quota. Please check your subscription and billing details."}, None
        return None, {"Error": "An unexpected error occurred. Please try again later."}, None



# Final layer to let user talk about the test and query his doubts
def chat(user_prompt: str, data_context_json: Dict[str, Any], llm_response: Dict[str, Any], test_results: Dict[str, Any], plot_context: Dict[str, Any], user_chat: str, chat_context: list):
    
    chat_history = render_chat(chat_context)

    try:
        for chunks in chat_chain.stream({
        'user_prompt': user_prompt,
        'data_context': data_context_json,
        'llm_response': llm_response,
        'test_results': test_results,
        'plot_context': plot_context,
        'user_chat': user_chat,
        'chat_history': chat_history

    }):
            print(chunks.content, end='', flush=True)
            yield chunks.content

    except ResourceExhausted:
        return None, {'Error:': f"You have exhausted the free Token. Please try again later!"}, None
    
    except Exception as e:
        # Fallback in case you can't catch ResourceExhausted explicitly
        if "ResourceExhausted" in str(e) or "quota" in str(e).lower():
            return None, {"Error": "You have exceeded your quota. Please check your subscription and billing details."}, None
        return None, {"Error": "An unexpected error occurred. Please try again later."}, None
