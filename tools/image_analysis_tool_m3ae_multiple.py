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
from utils import correct_malformed_json
from vqa_m3ae import post_vqa_m3ae_with_url

from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)

# 'For example, 1. text2SQL("given the last study of patient 13859433 this year") and then 2. image_analysis("are there any anatomicalfinding that are still no in the left hilar structures in $1") is NEVER allowed. '
#'Use 2. image_analysis("are there any anatomicalfinding that are still no in the left hilar structures", context=["$1"]) instead.\n'
_DESCRIPTION = (
    " image_analysis(question:str, context: Union[str, List[str]])-> str\n"
    " This tools is a medical image analysis task. For given radiology images and a question in English, it analysis the images and provide answer to the question. \n"
    " The given question should be in English. It it is in other language you should translate it to English."
    " Comparision should be done after each analysis.\n"
    "- You cannot analyse multiple studies in one call. For instance, `image_analysis('are there any anatomical finding that are still absent in the left hilar structures?','[{{'image_id':xxx,'stuy_id':yyy}}, {{'image_id':zzz,'stuy_id':www}})` does not work. "
    "If you need to analyse multiple images, you need to call them separately like `image_analysis('are there any anatomical finding that are still absent in the left hilar structures?','{{'image_id':xxx,'stuy_id':yyy}}')` and then `image_analysis('are there any anatomical finding that are still absent in the left hilar structures?','{{'image_id':zzz,'stuy_id':wwww}}')`\n"
    "These are the samples and you should consider the give question and act accordingly. "
    " - Minimize the number of `image_analysis` actions as much as possible."
    # Context specific rules below
    " - You should provide either list of strings or string as `context` from previous agent to help the `image analysis` agent solve the problem. "
    "If there are multiple contexts you need to answer the question, you can provide them as a list of strings.\n"
    " - `image_analysis` action will not see the output of the previous actions unless you provide it as `context`. "
    "You MUST provide the output of the previous actions as `context` if you need to do image_analysis on it.\n"
    " - You MUST NEVER provide `text2SQL` type action's outputs as a variable in the `question` argument. "
    "This is because `text2SQL` returns a text blob that contains the information about the database record, and needs to be process and extract image_id which `image_analysis` requires "
    "Therefore, when you need to provide an output of `text2SQL` action, you MUST provide it as a `context` argument to `image_analysis` action. "
)


# _SYSTEM_PROMPT = """You are a medical image analysis assistant. Analyze the the provided question and image to answer the question.
# """

# _ADDITIONAL_CONTEXT_PROMPT = """The following additional context is provided from other functions.\
#     Use it to substitute into any ${{#}} variables or other words in the problem.\
#     \n\n${context}\n\nNote that context variables are not defined in code yet.\
# You must extract the relevant image_id and directly put them in code.
# """


def _get_study_id(data,db_path):
    try:
        where=[f"{k}='{v}'" for k,v in data.items() if v is not None]
        # print("where",where)
        query=f"SELECT study_id from TB_CXR where {' '.join(where)}"
        # print ("query where u dont have the study_id in origin",query)
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query)
        results = [dict(row) for row in cur.fetchall()]
        return results[-1]
    except:
        return None

def _get_image_url(_d, db_path, current_path ='.'):
     
    # print("_d",_d)
    if 'image_id' not in _d and 'study_id' not in _d:
            _d =_get_study_id(_d, db_path)
            if _get_study_id(_d, db_path) is None:
                return ValueError(f"The image analysis task requires image_id or study_id or any related in the data\nstate:\n{_d}")
            
    d = _d
    root_path = Path(current_path).resolve()
    files_path = root_path /'files' 
    res=[]
    if 'image_id' in d:
        # find all the image files under the folder
        
    #/home/ubuntu/workspace/XMODE-LLMCompiler/files/p15/p15833469/s57883509/1b0b0385-a72d064d-be1f11ed-a39331d1-dde8f464.jpg
        image_nanme = f"{d['image_id']}.jpg"
        d['image_url'] = [f for f in files_path.rglob(image_nanme)][0]
        res = [d]
    
    elif 'image_id' not in d and 'study_id' in d:
        # find all the folders to the name
        study_folder = f"s{d['study_id']}"
        # get the folder path
        study_folder_path = [f for f in files_path.rglob(study_folder) if f.is_dir()][0]
        # list all *.jpg files under the folder
        all_image_paths = list(study_folder_path.rglob('*.jpg'))
        res = [d] * len(all_image_paths)
        for d, img in zip(res, all_image_paths):
            d['image_id'] = img.stem
            d['image_url'] = img
    return res

def _load_image(image_url):
    try:
        image = base64.b64encode(image_url.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{image}"
    except FileNotFoundError:
        raise FileNotFoundError(f"Image_path <{image_url}> not found")
    except base64.binascii.Error:
        raise ValueError(f"Image_path <{image_url}> is not a valid image")
    except Exception as e:
        raise e

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


def get_image_analysis_tools(db_path:str):
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
        
        print("context-first:", context,type(context))
        # {'status': 'success', 
        # 'data': [{'subject_id': 10501557, 'study_id': 50494231, 'image_id': 'dc8e362f-c7eed9e4-23b6a482-f5914468-cfee1143', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p10/p10501557/s50494231/dc8e362f-c7eed9e4-23b6a482-f5914468-cfee1143.jpg'}, {'subject_id': 10501557, 'study_id': 52176984, 'image_id': 'b011d8cc-dc7132b2-88dbf1ce-25edfe98-e7f91d64', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p10/p10501557/s52176984/b011d8cc-dc7132b2-88dbf1ce-25edfe98-e7f91d64.jpg'}, {'subject_id': 10737408, 'study_id': 52883453, 'image_id': 'b67361c3-3f5ae62e-460f6431-325adf4d-0d2b1e14', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p10/p10737408/s52883453/b67361c3-3f5ae62e-460f6431-325adf4d-0d2b1e14.jpg'}, {'subject_id': 10737408, 'study_id': 57292936, 'image_id': '42129ce4-60759151-2961e757-ed99e1bc-5fbd5220', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p10/p10737408/s57292936/42129ce4-60759151-2961e757-ed99e1bc-5fbd5220.jpg'}, {'subject_id': 11053589, 'study_id': 56374336, 'image_id': '3ff536d0-58474ed7-f99f6fc1-54ca116d-4838ad24', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p11/p11053589/s56374336/3ff536d0-58474ed7-f99f6fc1-54ca116d-4838ad24.jpg'}, {'subject_id': 12215941, 'study_id': 50754556, 'image_id': '3c2dc8ec-12dcbe00-cb056bd4-ea8a493b-c53906b9', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p12/p12215941/s50754556/3c2dc8ec-12dcbe00-cb056bd4-ea8a493b-c53906b9.jpg'}, {'subject_id': 12215941, 'study_id': 52085431, 'image_id': '1212710f-946955fb-af038081-0e3072dd-4cd7f187', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p12/p12215941/s52085431/1212710f-946955fb-af038081-0e3072dd-4cd7f187.jpg'}, {'subject_id': 12215941, 'study_id': 53288444, 'image_id': '320613f2-ebd6f20f-3a56947b-9708596e-d18cbc9c', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p12/p12215941/s53288444/320613f2-ebd6f20f-3a56947b-9708596e-d18cbc9c.jpg'}, {'subject_id': 12215941, 'study_id': 53532266, 'image_id': '2aee3eca-ddf6adea-80e26b80-ff12f81b-46dc2827', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p12/p12215941/s53532266/2aee3eca-ddf6adea-80e26b80-ff12f81b-46dc2827.jpg'}, {'subject_id': 12215941, 'study_id': 55608075, 'image_id': 'fca2a1d0-e0ec7afb-45e9bb8a-da768304-f436b847', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p12/p12215941/s55608075/fca2a1d0-e0ec7afb-45e9bb8a-da768304-f436b847.jpg'}, {'subject_id': 12215941, 'study_id': 56249693, 'image_id': '32a9152c-436bb18e-69e122a1-662efda0-9666c777', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p12/p12215941/s56249693/32a9152c-436bb18e-69e122a1-662efda0-9666c777.jpg'}, {'subject_id': 12215941, 'study_id': 57437645, 'image_id': '4d7402d2-082bca91-c0a0d1b8-6604563f-9ebc18c8', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p12/p12215941/s57437645/4d7402d2-082bca91-c0a0d1b8-6604563f-9ebc18c8.jpg'}, {'subject_id': 12215941, 'study_id': 59403367, 'image_id': '2a0efdc6-fb6184c1-eb1e1224-44130630-a728b892', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p12/p12215941/s59403367/2a0efdc6-fb6184c1-eb1e1224-44130630-a728b892.jpg'}, {'subject_id': 12706984, 'study_id': 51503517, 'image_id': '676ac094-48152b11-2b1fcd07-7abfc405-fa272730', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p12/p12706984/s51503517/676ac094-48152b11-2b1fcd07-7abfc405-fa272730.jpg'}, {'subject_id': 12706984, 'study_id': 52466701, 'image_id': '6cf1d183-15cbad33-6b9854b9-43458140-0f51c28d', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p12/p12706984/s52466701/6cf1d183-15cbad33-6b9854b9-43458140-0f51c28d.jpg'}, {'subject_id': 12706984, 'study_id': 54561293, 'image_id': '86a893f4-a3c086cf-c5136df2-262748f2-60d34cd7', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p12/p12706984/s54561293/86a893f4-a3c086cf-c5136df2-262748f2-60d34cd7.jpg'}, {'subject_id': 13399590, 'study_id': 52493066, 'image_id': 'd4c0d31b-d651ced4-d775a38d-3572ffab-09bdcc55', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p13/p13399590/s52493066/d4c0d31b-d651ced4-d775a38d-3572ffab-09bdcc55.jpg'}, {'subject_id': 15655083, 'study_id': 52961339, 'image_id': 'b41a99e3-b99c48ff-c484796b-06aeeae5-bee45053', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p15/p15655083/s52961339/b41a99e3-b99c48ff-c484796b-06aeeae5-bee45053.jpg'}, {'subject_id': 15655083, 'study_id': 59443706, 'image_id': '8384b7b2-3d508895-e46f209d-3479d8cd-4d0bff58', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p15/p15655083/s59443706/8384b7b2-3d508895-e46f209d-3479d8cd-4d0bff58.jpg'}, {'subject_id': 16015533, 'study_id': 59937280, 'image_id': '12e31810-ee29a935-d370c78c-647f1a93-6711d17b', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p16/p16015533/s59937280/12e31810-ee29a935-d370c78c-647f1a93-6711d17b.jpg'}, {'subject_id': 16191519, 'study_id': 52376941, 'image_id': '16566e1d-a19c8451-c902faa7-38be8665-aad81289', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p16/p16191519/s52376941/16566e1d-a19c8451-c902faa7-38be8665-aad81289.jpg'}, {'subject_id': 16863257, 'study_id': 54933394, 'image_id': '80b5e62a-6b7afec0-66ef3b70-65071be5-6fe3de96', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p16/p16863257/s54933394/80b5e62a-6b7afec0-66ef3b70-65071be5-6fe3de96.jpg'}, {'subject_id': 16905933, 'study_id': 50271633, 'image_id': '79d054e6-2c13022c-6e82bdfa-a24015ca-045ae289', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p16/p16905933/s50271633/79d054e6-2c13022c-6e82bdfa-a24015ca-045ae289.jpg'}, {'subject_id': 16905933, 'study_id': 53951212, 'image_id': '65a2f615-3d2088b8-e1297f47-cab1c5b7-c7482bde', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p16/p16905933/s53951212/65a2f615-3d2088b8-e1297f47-cab1c5b7-c7482bde.jpg'}, {'subject_id': 16905933, 'study_id': 57571812, 'image_id': '997a7b0a-a1c6567a-2a411fd4-268667f7-a4a6f8cd', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p16/p16905933/s57571812/997a7b0a-a1c6567a-2a411fd4-268667f7-a4a6f8cd.jpg'}, {'subject_id': 18002691, 'study_id': 53305998, 'image_id': 'dd21aa73-06598e56-e1a64eb1-16f59abf-4f0e5b47', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p18/p18002691/s53305998/dd21aa73-06598e56-e1a64eb1-16f59abf-4f0e5b47.jpg'}, {'subject_id': 19243401, 'study_id': 55804087, 'image_id': '2828abaf-4507dd27-6badf4ec-a5165223-42866eb0', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p19/p19243401/s55804087/2828abaf-4507dd27-6badf4ec-a5165223-42866eb0.jpg'}, {'subject_id': 19243401, 'study_id': 57101156, 'image_id': '635fcd1f-11af2a65-c6d165a9-fc6822d2-560fd246', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p19/p19243401/s57101156/635fcd1f-11af2a65-c6d165a9-fc6822d2-560fd246.jpg'}, {'subject_id': 19945642, 'study_id': 51880304, 'image_id': '1791b2c7-36a45e03-4b55f61f-72fa83eb-f08952dc', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p19/p19945642/s51880304/1791b2c7-36a45e03-4b55f61f-72fa83eb-f08952dc.jpg'}, {'subject_id': 19945642, 'study_id': 53737542, 'image_id': 'f60c0379-7ea41720-8cb36a47-ede6cb5d-8e5e4ff2', 'image_path': '/home/ubuntu/workspace/XMODE-LLMCompiler/files/p19/p19945642/s53737542/f60c0379-7ea41720-8cb36a47-ede6cb5d-8e5e4ff2.jpg'}]} 
        # <class 'str'>

      
        if isinstance(context, str):
            context=correct_malformed_json(context)
            context = [ast.literal_eval(context)]
            if 'status' in context[0]:
                context=context[0]
        else:
            print("context-2", context)
            context = ast.literal_eval(context[0])
        
        print("context-2", context)
            # If the context contains 'data' key, use its value
        if 'data' in context:
            #["{'status': 'success', 'data': [{'studydatetime': '2105-09-06 18:18:18'}]}"]
            context = context['data']

        print("context-after:", context)
        if not isinstance(context, list):
            print("context-after in not list", list(context))
            context=[context]
        try:
            
            if len(context)>1:
                vqa_answers=[]
                for ctx in context:
                    chain_input = {"question": question}
                    image_url=Path(ctx['image_path'])
                    images_encoded=_load_image(image_url)
                    vqa_answer=post_vqa_m3ae_with_url(chain_input["question"],images_encoded)
                    ctx[question]=vqa_answer['vqa_answers']
                    vqa_answers.append(ctx)
                print(vqa_answers)
                return vqa_answers
            else:
                chain_input = {"question": question}
                image_url=Path(ctx['image_path'])
                images_encoded=_load_image(image_url)
                vqa_answer=post_vqa_m3ae_with_url(chain_input["question"],images_encoded)
                return 
        
        except Exception as e:
            return repr(e)

    return StructuredTool.from_function(
        name="image_analysis",
        func=image_analysis,
        description=_DESCRIPTION,
    )
