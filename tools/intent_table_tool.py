from typing import List, Dict, Any, Optional
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from src.utils import _get_db_schema_json, _get_db_schema
import sqlite3
import json

# @TODO test the code alone and with the text2SQL tool

_DESCRIPTION = (
    "intent_tables(problem: str, context:str) --> List[str]\n"
    "The input for this tool should be `problem` as a textual question and `context` as a database_schema.\n"
    "This tool is able to identify and return a list of the tables in the database schema that are relevant for solving the given problem."
)

_SYSTEM_PROMPT = """
You are a database expert. Your task is to identify which tables in a database schema are relevant to a user query. This tool MUST BE APPLIED before any text2SQL tool to identify the relevant tables for the query.
Given the user question and the database schema, analyze the query's intent, and return a list of the table names that are necessary to answer the question.

Guidelines:
- Carefully consider the query's specific requirements, such as filtering, aggregation, or relations between tables.
- Include only the tables that are directly involved in generating the result for the user question.
- If the question implies joining or filtering across multiple tables, list all involved tables.
- If the question cannot be answered using the schema, or no tables are relevant, return an empty list.
"""

# Question: {problem}
# Schema: {db_schema}
# Answer: ${{Output}}

# Begin.

# Question: Retrieve the last study of a patient this year.
# Schema: Schema: {{"TB_CXR": {{"COL": [["subject_id", "INTEGER"], ["studydatetime", "TEXT"]], "FK": [["TB_CXR.subject_id", "TB_PATIENTS.subject_id"]]}}, {{"TB_PATIENTS": {{"COL": [["subject_id", "INTEGER"], ["patient_name", "TEXT"]], "FK": []}}}}}}
# Answer: {{["TB_CXR"]}}

# Question: What are the details of patients with studies in the last month?
# Schema: Schema: {{"TB_CXR": {{"COL": [["subject_id", "INTEGER"], ["studydatetime", "TEXT"]], "FK": [["TB_CXR.subject_id", "TB_PATIENTS.subject_id"]]}}, {{"TB_PATIENTS": {{"COL": [["subject_id", "INTEGER"], ["patient_name", "TEXT"]], "FK": []}}}}}}
# Answer: {{["TB_CXR", "TB_PATIENTS"]}}

# Question: How many studies were performed this year?
# Schema: Schema: {{"TB_CXR": {{"COL": [["subject_id", "INTEGER"], ["studydatetime", "TEXT"]], "FK": [["TB_CXR.subject_id", "TB_PATIENTS.subject_id"]]}}, {{"TB_PATIENTS": {{"COL": [["subject_id", "INTEGER"], ["patient_name", "TEXT"]], "FK": []}}}}}}
# Answer: {{["TB_CXR"]}}

# If the database schema is large and has unrelated tables, ensure you include only the relevant ones. 
# Output the list of relevant table names as JSON.
class TableSelectionOutput(BaseModel):
   
    reasoning: str = Field(
        ...,
        description="The reasoning behind the extracted table names, including how context is included, if applicable.",
    )

    tables: List[str] = Field(
        ...,
        description="The list of tables from the database schema that are relevant to answering the user's question.",
    )

def get_intent_tables_tool(llm: ChatOpenAI, db_path: str):
    """
    Tool for identifying relevant tables in a database schema for a given question.

    Args:
        raw_question (str): The raw user question.
        schema (str): The database information such as the Tables and Columns.

    Returns:
        StructuredTool: The tool for identifying relevant tables.'
    """
    db_schema = _get_db_schema_json(db_path)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("user", "{problem}"),
            ("user", "{db_schema}"),
        ]
    )
    
    # extractor = create_structured_output_runnable(TableSelectionOutput, llm, prompt)
    extractor= prompt | llm.with_structured_output(TableSelectionOutput)

    
    
    
    def intent_tables(
        problem: str,
        context: Optional[str] = None,
    ):
        
        chain_input = {"problem": problem,}
        print("chain_input:", chain_input)
        try:
            result = extractor.invoke(chain_input)
            # print("result:", result)
            tables =  result.tables
            return tables
        except Exception as e:
            return {"error": str(e)}

    return StructuredTool.from_function(
        name="intent_tables",
        func=intent_tables,
        description=_DESCRIPTION,
    )