
import re
from typing import List, Optional
import json
import numexpr
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI


_DESCRIPTION = (
    "columns_identifier(problem: str, context: List[str])-> dict\n"
    " This tools Identifies relevant columns from the given database schema for a specific question. \n"
    " If some of the required information is not directly available in the schema provided, It adds IMAGE_ID to the list of columns.\n"
    " It concats the table name to the column name like TABLE.COLUMN\n"
)


_SYSTEM_PROMPT = """Identify relevant columns from the given database schema for a given question.  Dont provide extra information.
If some of the required information is not directly available in the schema provided, it adds IMAGE_ID to the list of columns.
Add the table name to the column name like TABLE_NAME.COLUMN_NAME.
"""

_ADDITIONAL_CONTEXT_PROMPT = """ Database Schema
CREATE TABLE TB_CXR
        (
            ROW_ID INT NOT NULL,
            SUBJECT_ID INT NOT NULL,
            HADM_ID INT,
            STUDY_ID INT NOT NULL,
            IMAGE_ID INT NOT NULL,
            STUDYDATETIME TIMESTAMP(0) NOT NULL,
            VIEWPOSITION VARCHAR(20) NOT NULL,
            STUDYORDER INT NOT NULL,
            CONSTRAINT tb_cxr_rowid_pk PRIMARY KEY (ROW_ID)
        )
"""


class ExecuteCode(BaseModel):

    reasoning: str = Field(
        ...,
        description="The reasoning behind the columns extraction, including how context is included, if applicable.",
    )

    columns: list[str] = Field(
        ...,
        description="list of columns ",
    )
    tables: list[str] = Field(
        ...,
        description="list of tables ",
    )


output_schema = {
                    "type": "function",
                    "function": {
                        "name": "columns_identifier",
                        "description": "",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "tables": {
                                    "description": "list of tables",
                                    "type": "string"
                                },
                                "columns": {
                                    "description": "list of columns",
                                    "type": "string"
                                },
                                "reasoning": {
                                    "description": "The reasoning behind the columns extraction, including how context is included, if applicable.",
                                    "type": "string"
                                }
                            },
                            "required": ["tables", "columns"]
                        }
                    }
                }

def get_identifier_tools(llm: ChatOpenAI):
    """
    Identifies relevant columns from the given database schema.

    Args:
        raw_question (str): The raw user question.
        schema (dict): The database schema.
        api_key (str): The OpenAI API key.

    Returns:
        dict: The relevant columns and their tables.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("user", "{problem}"),
            MessagesPlaceholder(variable_name="context", optional=True),
        ]
    )

    extractor = create_structured_output_runnable(
        output_schema,
        llm, 
        prompt,
        mode="openai-tools",
        enforce_function_usage=True, 
        return_single=True)

    def identify_columns(
        problem: str,
        context: Optional[List[str]]= None,
        config: Optional[RunnableConfig] = None,
    ):
        chain_input = {"problem": problem}
       
        if context:
            context_str = "\n".join(context)
            if context_str.strip():
                context_str = _ADDITIONAL_CONTEXT_PROMPT.format(
                    context=context_str.strip()
                )
                chain_input["context"] = [SystemMessage(content=context_str)]
       
        code_model = extractor.invoke(chain_input, config)
        try:
            return  code_model
        except Exception as e:
            return repr(e)

    return StructuredTool.from_function(
        name="columns_identifier",
        func=identify_columns,
        description=_DESCRIPTION,
    )

