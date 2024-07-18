import re
from typing import List, Optional
import json
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

from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)


_DESCRIPTION = (
    " data_analysis(question:str, context: Optional[list[str]])-> str\n"
    " This tools is a medical data analysis task. For given data and a question, it analysis the data and provide answer to the question. \n"
    " Comparision should be done after each analysis.\n"
    "- You cannot analyse multiple studies in one call. For instance, `image_analysis('are there any anatomicalfinding that are still no in the left hilar structures?','[{{'image_id':xxx,'stuy_id':yyy}}, {{'image_id':zzz,'stuy_id':www}})` does not work. "
    "If you need to analyse multiple images, you need to call them separately like `image_analysis('are there any anatomicalfinding that are still no in the left hilar structures?','{{'image_id':xxx,'stuy_id':yyy}}')` and then `image_analysis('are there any anatomicalfinding that are still no in the left hilar structures?','{{'image_id':zzz,'stuy_id':wwww}}')`\n"
    "These are the samples and you should consider the give question and act accordingly. "
    " - Minimize the number of `image_analysis` actions as much as possible."
    # Context specific rules below
    " - You can optionally provide a list of strings as `context` to help the agent solve the problem. "
    "If there are multiple contexts you need to answer the question, you can provide them as a list of strings.\n"
    " - `image_analysis` action will not see the output of the previous actions unless you provide it as `context`. "
    "You MUST provide the output of the previous actions as `context` if you need to do image_analysis on it.\n"
    " - You MUST NEVER provide `text2SQL` type action's outputs as a variable in the `question` argument. "
    "This is because `text2SQL` returns a text blob that contains the information about the database record, and needs to be process and extract image_id which `image_analysis` requires "
    "Therefore, when you need to provide an output of `text2SQL` action, you MUST provide it as a `context` argument to `image_analysis` action. "
)


_SYSTEM_PROMPT = """You are a medical image analysis assistant. Analyze the the provided question and image to answer the question.
"""

_ADDITIONAL_CONTEXT_PROMPT = """The following additional context is provided from other functions.\
    Use it to substitute into any ${{#}} variables or other words in the problem.\
    \n\n${context}\n\nNote that context variables are not defined in code yet.\
You must extract the relevant image_id and directly put them in code.
"""

def _get_image_url(_d, current_path ='.'):
    if 'image_id' not in _d and 'study_id' not in _d:
            raise ValueError(f"The image analysis task requires image_id or study_id in the data\nstate:\n{_d}")
    d = _d
    root_path = Path(current_path).resolve()
    files_path = root_path /'files' 
    res=[]
    if 'image_id' in d:
        # find all the image files under the folder
        image_nanme = f"{d['image_id']}.jpg"
        d['image_url'] = [f for f in files_path.rglob(image_nanme) if f.is_file()][0].as_posix()
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
            d['image_url'] = img.as_posix()
    return res

def _load_image(image_url):
   
    if image_url.startswith("http"):
        return image_url
    else:
        try:
            with open(image_url, "rb") as image:
                image = base64.b64encode(image.read()).decode("utf-8")
                return f"data:image/jpeg;base64,{image}"
        except FileNotFoundError:
            raise FileNotFoundError(f"Image_path <{image_url}> not found")
        except base64.binascii.Error:
            raise ValueError(f"Image_path <{image_url}> is not a valid image")
        except Exception as e:
            raise e
  
class ExecuteCode(BaseModel):

    reasoning: str = Field(
        ...,
        description="The reasoning behind the answer, including how context is included, if applicable.",
    )

    answer: str = Field(
        ...,
        description="an answer to the question about the image",
    )
   


def get_image_analysis_tools(llm: ChatOpenAI):
    """
   
    Args:
        question (str): The question about the image.
        context list(str)
    Returns:
        str: the answer to the question about the image.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("user", "{question}"),
            MessagesPlaceholder(variable_name="image_info"),
        ]
    )
    extractor = create_structured_output_runnable(ExecuteCode, llm, prompt)

    

    def image_analysis(
        question: str,
        context: list[Optional[str]] = None,
        config: Optional[RunnableConfig] = None,
    ):
        chain_input = {"question": question}
        #test
        
        #context="[{'study_id': 56222792, 'image_id': '3c7d71b0-383da7fc-80f78f8c-6be2da46-3614e059'}]"
        if context :
            print("context",context,type(context))
            # if context_str.strip():
            #     context_str = _ADDITIONAL_CONTEXT_PROMPT.format(
            #         context=context_str.strip()
            #     )
            # If context is a string, parse it as JSON
            context = context[-1]
            if isinstance(context, str):
                context = json.loads(context.replace("'", "\""))
                # If the context contains 'data' key, use its value
            else:
                 context = json.loads(context)
            if 'data' in context:
                context = context['data']

            image_urls = [_get_image_url(ctx) for ctx in context ]
            images_encoded = [_load_image(url['image_url']) for url in image_urls ]

            _humMessage=[{"type": "image_url", "image_url":{"url":x,"detail": "low"}} for x in images_encoded]
            print(_humMessage)
            chain_input["image_info"] = [
                HumanMessage(
                    content=_humMessage
                )
            ]
                       
        model = extractor.invoke(chain_input, config)
        try:
            return model.answer
        except Exception as e:
            return repr(e)

    return StructuredTool.from_function(
        name="image_analysis",
        func=image_analysis,
        description=_DESCRIPTION,
    )

