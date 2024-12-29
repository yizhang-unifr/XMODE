import re, sys,os
sys.path.append(os.path.dirname(os.getcwd()) + '/src')
sys.path.append(os.path.dirname(os.getcwd()) + '/tools')

from typing import List, Optional, Union
import json
import ast

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from PIL import Image
import base64
from pathlib import Path
from src.utils import correct_malformed_json





IMAGE_PATH='/home/ubuntu/workspace/XMODE/ArtWork/data'
# 'For example, 1. text2SQL("given the last study of patient 13859433 this year") and then 2. image_analysis("are there any anatomicalfinding that are still no in the left hilar structures in $1") is NEVER allowed. '
#'Use 2. image_analysis("are there any anatomicalfinding that are still no in the left hilar structures", context=["$1"]) instead.\n'
_DESCRIPTION = (
    " image_analysis(question:str, context: Union[str, List[str]])-> str\n"
    " This tools is an image analysis task. For given image and a question, it analysis the image and provide answer to the question. \n"
    " The question should target only one image. For example: is there object <X> in the image? or how many <X> appears in the image? or does image depicts <Y>?"
    " It is useful for when you want to know what is depicted on the image.\n"
    " The question can be anything that can be answered by looking at an image: For example. How many <x> are depicted? Is <y> depicted? What is in the background? ...\n"
    " Comparision should be done after each analysis.\n"
    " - Minimize the number of `image_analysis` actions as much as possible."
    # Context specific rules below
    " - You should provide either list of strings or string as `context` from previous agent to help the `image analysis` agent solve the problem."
    "The format of the context for image_analysis should be `[{'img_path': 'xxxx'}, {'img_path': 'yyyy'}, ...]`. For example for one image: `[{'img_path': 'images/img_0.jpg'}]"
    "If there are multiple contexts you need to answer the question, you can provide them as a list of strings.\n"
    " - `image_analysis` action will not see the output of the previous actions unless you provide it as `context`. "
    "You MUST provide the output of the previous actions as `context` if you need to do image_analysis on it.\n"
    " - You MUST NEVER provide `text2SQL` type action's outputs as a variable in the `question` argument. "
    "This is because `text2SQL` returns a text blob that contains the information about the database record, and needs to be process and extract image_id which `image_analysis` requires "
    "Therefore, when you need to provide an output of `text2SQL` action, you MUST provide it as a `context` argument to `image_analysis` action. "
)



def extract_data(context):
    import re
    # Define the regex pattern to match the 'data' key and its value
    pattern = r"'data': (\[.*?\])\}"

    # Use re.search to find the 'data' value in the string
    match = re.search(pattern, context)

    if match:
        # Extract the matched value
        data_value = match.group(1)
        return data_value
    else:
        return None


def get_image_analysis_tools(vqa):
    """
   
    Args:
        question (str): The question about the image.
        context Union[str, List[str]]
    Returns:
        str: the answer to the question about the image.
    """

    def image_analysis(
        question: str,
        context: Union[str, List[str]],
    ):
        
        # print("context-first:", context,type(context))
      
        if isinstance(context, str):
            context=correct_malformed_json(context)
            
            context = [ast.literal_eval(context)]
            
            if 'status' in context[0]:
                context=context[0]
        else:
            # print("context-2", context)
            context = ast.literal_eval(context[0])
        
        # print("context-2", context)
        if 'data' in context:
            context = context['data']

        # print("context-after:", context)
        if not isinstance(context, list):
            # print("context-after in not list", [context])
            context=[context]
        try:
            vqa_answers=[]
            image_paths=[f"{IMAGE_PATH}/{ctx['img_path']}" for ctx in context]
            
            # print(image_paths)
            
            answers= vqa.extract(image_paths, question)
            
            for ctx , answer in zip(context,answers):
                ctx[question]=answer
                vqa_answers.append(ctx)
           
            return vqa_answers
        
        except Exception as e:
            return repr(e)

    return StructuredTool.from_function(
        name="image_analysis",
        func=image_analysis,
        description=_DESCRIPTION,
    )
