import sys
import os
print(sys.executable)
#os.environ['GOOGLE_API_KEY'] ='AIzaSyBPPjs__gGR7wAugqDf0xZftBn7-mTFbgs'
from langchain import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import google.generativeai as genai
import json
import pandas as pd
import traceback
import PyPDF2
from dotenv import load_dotenv
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

load_dotenv()

api_key=os.getenv("GOOGLE_API_KEY")

generation_config = {
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }

safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }

from langchain_google_genai import ChatGoogleGenerativeAI
genai.configure(api_key=api_key)
llm = ChatGoogleGenerativeAI(model="gemini-pro",
                 temperature=0.3)
     

TEMPLATE="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}

"""

quiz_generation_prompt=PromptTemplate(
    input_variables=["text","number","subject","tone","response_json"],
    template=TEMPLATE
)

quiz_chain=LLMChain(llm=llm,prompt=quiz_generation_prompt,output_key="quiz",verbose=True)

TEMPLATE2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of teh question and give a complete analysis of the quiz if the students
will be able to unserstand the questions and answer them. Only use at max 50 words for complexity analysis. 
if the quiz is not at par with the cognitive and analytical abilities of the students,\
update tech quiz questions which needs to be changed  and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt=PromptTemplate(
    input_variables=["subject","quiz"],
    template=TEMPLATE2
)

review_chain=LLMChain(llm=llm,prompt=quiz_evaluation_prompt,output_key="review",verbose=True)

generate_evaluation_chain=SequentialChain(chains=[quiz_chain,review_chain],input_variables=["text","subject","tone","number","response_json"],output_variables=["quiz","review"],verbose=True)

