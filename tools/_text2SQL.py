from typing import List, Optional
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
import sqlite3
import re
import json
from langchain.sql_database import SQLDatabase
from sqlalchemy import create_engine

from typing import Any, Callable, Dict, Literal, Optional, Sequence, Type, Union,List
_DESCRIPTION = (
    "text2SQL(problem: str, context:str)-->str\n"
    "The input for this tools should be `problem` as a textual question and `context` as a database_schema.\n"
    "This tools is able to translate the question to the SQL code considering the database information.\n"
    "The SQL code can be executed using sqlite3 library.\n"
    "Use the output of running generated SQL code to answer the question.\n"
)


_SYSTEM_PROMPT = """ 
You are a database expert. Generate a SQL query given the following user question and database information.
You should analyse the question and the database schema and come with the executabel sqlite3 query. 
Provide all the required information in the SQL code to answer the original user question that may required in other tasks utilizing the relevant database schema . Ensure you include all necessary information, including columns used for filtering, especially when the task involves plotting or data exploration.
It is also important to note that in the database, the current time is set to 2105-12-31 23:59:00 there if you have a question that ask about current time, set the current time equal to 2105-12-31 23:59:00. 
This must be taken into account when performing any time-based data queries or analyses.
If you want to consider "now" or "current_time", then replace them with `strftime('2105-12-31 23:59:59')`.
In the database genger `male` represented by `m' in lower case and `female` represent in `f` lower case.
if the question asks for specific abnormality, disease or finding in the paitent study and the relevant column (e.g., abnormality, findings) is not found in the database schema, you must retrieve the `image_path` of the study for image analysis.
Translate a text question into a SQL query that can be executed on the SQLite database. Use the output of running this SQL code to answer the question in the format: [{{"xxx": xxx, "yyy": yyy}}].

Question: ${{problem}}
ExecuteCode({{"SQL":"${{single line SQL code that answers the question considering the database schema information}}"}})
...execute_sql_query("{{SQL}}", "{{db_path}}")...
```output
${{Output of running the SQL query on db}}
```
Answer: ${{Output}}

Begin.

Question: Retrieve the last study of patient 13859433 this year.
ExecuteCode({{"SQL":"SELECT * FROM TB_CXR WHERE subject_id = 13859433 AND strftime('%Y', studydatetime) = '2105' ORDER BY studydatetime DESC LIMIT 1 "}})
...execute_sql_query("SELECT * FROM TB_CXR WHERE subject_id = 13859433 AND strftime('%Y', studydatetime) = '2105' ORDER BY studydatetime DESC LIMIT 1", "mimic_iv_cxr.db")...
```output
[{{"row_id": 651,"subject_id": 13859433, "hadm_id": 21193500, "study_id": 56222792, "image_path": "3files/p10/p10501557/s52176984/b011d8cc-dc7132b2-88dbf1ce-25edfe98-e7f91d64.jpg", "viewposition": "ap", "studydatetime": "2105-11-15 07:49:33}}]
```
Answer: [{{"row_id": 651,"subject_id": 13859433, "hadm_id": 21193500, "study_id": 56222792, "image_path": "/files/p10/p10501557/s50494231/dc8e362f-c7eed9e4-23b6a482-f5914468-cfee1143.jpg", "viewposition": "ap", "studydatetime": "2105-11-15 07:49:33}}]

Question: Retrieve the study before the last for patient 13859433 this year.
ExecuteCode({{"SQL": "SELECT study_id FROM ( SELECT study_id, ROW_NUMBER() OVER (ORDER BY studydatetime DESC) AS rk FROM TB_CXR WHERE subject_id = 13859433 AND strftime('%Y', studydatetime) = '2105' ) WHERE rk = 2"}})
...execute_sql_query("SELECT study_id FROM ( SELECT study_id, ROW_NUMBER() OVER (ORDER BY studydatetime DESC) AS rk FROM TB_CXR WHERE subject_id = 13859433 AND strftime('%Y', studydatetime) = '2105' ) WHERE rk = 2", "mimic_iv_cxr.db")...
```output
[{{"study_id": 51083001}}]
```
Answer: [{{"study_id": 51083001}}]

"""
#If you want to consider "now" or "current_time", then replace them with strftime('2105-12-31 23:59:59').
_ADDITIONAL_CONTEXT_PROMPT = """
"""


class ExecuteCode(BaseModel):

    reasoning: str = Field(
        ...,
        description="The reasoning behind the SQL expression, including how context is included, if applicable.",
    )

    SQL: str = Field(
        ...,
        description="The SQL Code that can be runnable on the corresponding database ",
    )
    

# def _clean_query(q):
#     res=q.replace('\n'," ").replace('%y','%Y').replace('current_time',"strftime('2105-12-31 23:59:59')").replace('now',"strftime('2105-12-31 23:59:59')")
#     return res


# def _parse_input(input_text: str) -> dict:
#     """
#     Args:
#         input_text (str): The text containing input in dict format.

#     Returns:
#         dict: The parsed sub-questions as a structured dictionary.
#     """
#     try:
#         # Remove any potential non-JSON text and extract the JSON part
#         json_match = re.search(r'\{.*\}', input_text, re.DOTALL)
#         if json_match:
#             input_text_json = json_match.group(0)
#             input_text = json.loads(input_text_json)
#         else:
#             raise ValueError("No JSON object found in the text")
#         return input_text
#     except (json.JSONDecodeError, ValueError) as e:
#         print(f"Error parsing sub-questions: {e}")
#         return {
#             "tables":[],
#             "columns":[]
#         }
    
def _execute_sql_query(query: str, db_path: str, as_dict=True) -> Dict[str, Any]:
    def change_current_time(q):
        res = q.replace(
            "current_time", "strftime('2105-12-31 23:59:59')"
        ).replace(
            "now", "strftime('2105-12-31 23:59:59')"
        ).replace(
            '%y','%Y'
        ). replace('\n'," ")
        
        return res
    
    try:
        if as_dict:
            import sqlite3
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            # print("SQL:",change_current_time(query))
            cur.execute(change_current_time(query))
            results = [dict(row) for row in cur.fetchall()]
            # print("results of SQL",results)
        else:
            engine = create_engine(f'sqlite:///{db_path}')
            database = SQLDatabase(engine, sample_rows_in_table_info=0)
            results = database.run(change_current_time(query))
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}



def get_text2SQL_tools(llm: ChatOpenAI, db_path:str):
    """
    Provide the SQL code from a given question.

    Args:
        raw_question (str): The raw user question.
        schema (str): The database information such as the Tables and Columns.
        api_key (str): The OpenAI API key.

    Returns:
        results (str)
    """

    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("user", "{problem}"),
            ("user", "{db_schema}"),
           
        ]
    )
    extractor = create_structured_output_runnable(ExecuteCode, llm, prompt)

    def text2SQL(
        problem: str,
        context:str,
        config: Optional[RunnableConfig] = None,
    ):
        #tables_columns=_parse_input(tables_columns)
        chain_input = {"problem": problem,"db_schema":context}
        code_model = extractor.invoke(chain_input, config)
        try:
            return _execute_sql_query(code_model.SQL, db_path)
        except Exception as e:
            return repr(e)

    return StructuredTool.from_function(
        name="text2SQL",
        func=text2SQL,
        description=_DESCRIPTION,
    )

