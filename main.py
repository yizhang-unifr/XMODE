import getpass
import os

from langchain_openai import ChatOpenAI

# Imported from the https://github.com/langchain-ai/langgraph/tree/main/examples/plan-and-execute repo
from tools.identify_columns import get_identifier_tools
from tools.text2SQL import get_text2SQL_tools
from tools.image_analysis_tool import get_image_analysis_tools
from langchain.sql_database import SQLDatabase
from sqlalchemy import create_engine

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from src.joiner import Replan, JoinOutputs
from src.joiner import *

from typing import Sequence

from langchain import hub
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

import itertools
from src.planner import *
from src.task_fetching_unit import *
from src.joiner import *
from src.joiner import parse_joiner_output
from src.utils import _get_db_schema
from typing import Dict
from src.utils import correct_malformed_json,timeout

from langgraph.graph import END, MessageGraph, START
import json
import ast

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

#OPENAI_API_KEY: sk-statbot-demo-GW8XbAyWKdjCapASEcuYT3BlbkFJcsmRiIGe4LeMsmcBkN4N
#LANGCHAIN_API_KEY: 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
#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
#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
_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("LANGCHAIN_API_KEY")
# _set_if_undefined("TAVILY_API_KEY")
# Optional, add tracing in LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "M3LX-ehrxqa-30"

@timeout(180)
def run_m3lx(question, tables, model):
    result={}
    db_path="/home/ubuntu/workspace/M3LX-LLMCompiler/mimic_iv_cxr.db"
    tables= [t.upper()for t in tables]
    temperature=0
    ## Tools
    database_schema =_get_db_schema(db_path, tables)

    translate= get_text2SQL_tools(ChatOpenAI(model=model, temperature=temperature),db_path)
    image_analysis=get_image_analysis_tools(ChatOpenAI(model="gpt-4o", temperature= temperature),db_path)
    tools = [translate,image_analysis]
    llm = ChatOpenAI(model=model)
    prompt = ChatPromptTemplate.from_messages(
    [
    ("system",'''Given a user question and a database schema, analyze the question to identify and break it down into relevant sub-questions. 
     Determine which tools (e.g., text2SQL, image_analysis, reasoning) are appropriate for answering each sub-question based on the available database information and tools.
     First, you should begin by thoroughly analyzing the user's main question. It’s important to understand the key components and objectives within the query.
     Next, you musr review the provided database schema. This involves examining the tables, fields, and relationships within the database to identify which parts of the schema are relevant to the user’s question, and creat a set of sub-questions.
     In cases where the user’s question involves data that is not directly available in the database schema —such as when there is no corresponding table or column for the required information— you must consider the need for image analysis. 
     For instance, if the question involves comparision of study for specific abnormality, disease or finding in the paitent study and the relevant column (e.g., abnormality, findings, chest x-ray) is not found in the database schema, you must retrieve th`image_id` of the study for image analysis, 
     This ensures we can address parts of the question that rely on visual data.
     With a clear understanding of the question and the database schema, you can now break down the main question into smaller, more manageable sub-questions. 
     These sub-questions should each target a specific aspect of the main question. 
     After identifying the sub-questions, you should determine the most appropriate tools to answer each one. Depending on the nature of the sub-questions, we might use a variety of tools.
     Each sub-question should be a textual question. Dont generate a code as a sub-question.
     Each image analysis task should consider only one study. Study id is different than image id. 
     Include the following database schema in the context for text2SQL tasks. 
     {database_schema}
     It is also important to note that in the database, the current time is set to 2105-12-31 23:59:00. This must be taken into account when performing any time-based data queries or analyses.
     In any database retreival task, retieve `subject_id` together with other columns.
     Create a plan to solve it with the utmost parallelizability. 
     Each plan should comprise an action from the following  {num_tools} types:
    {tool_descriptions}
    {num_tools}. join(): Collects and combines results from prior actions.

 - An LLM agent is called upon invoking join() to either finalize the user query or wait until the plans are executed.
 - join should always be the last action in the plan, and will be called in two scenarios:
   (a) if the answer can be determined by gathering the outputs from tasks to generate the final response.
   (b) if the answer cannot be determined in the planning phase before you execute the plans. Guidelines:
 - Each action described above contains input/output types and description.
    - You must strictly adhere to the input and output types for each action.
    - The action descriptions contain the guidelines. You MUST strictly follow those guidelines when you use the actions.
 - Each action in the plan should strictly be one of the above types. Follow the Python conventions for each action.
 - Each action MUST have a unique ID, which is strictly increasing.
 - Inputs for actions can either be constants or outputs from preceding actions. In the latter case, use the format $id to denote the ID of the previous action whose output will be the input.
 - If there is an input from from preceding actions, always point its id as `$id` in the context of the action/
 - Always call join as the last action in the plan. Say '<END_OF_PLAN>' after you call join
 - Ensure the plan maximizes parallelizability.
 - Only use the provided action types. If a query cannot be addressed using these, invoke the join action for the next steps.
 - Never introduce new actions other than the ones provided.'''),
        ("user", '{messages}'),
        ("assistant", 'Remember, ONLY respond with the task list in the correct format! E.g.:\nidx. tool(arg_name=args)'),

    ]
)
    # This is the primary "agent" in our application
    planner = create_planner(llm, tools, prompt, database_schema)
    #example_question = "is there evidence in the last study for patient 13859433 this year of any anatomical findings in the left hilar structures still absent compared to the previous study?"

    ## Task Fetching Unit
    @as_runnable
    def plan_and_schedule(messages: List[BaseMessage], config):
        tasks = planner.stream(messages, config)
        # Begin executing the planner immediately
        try:
            tasks = itertools.chain([next(tasks)], tasks)
        except StopIteration:
            # Handle the case where tasks is empty.
            tasks = iter([])
        scheduled_tasks = schedule_tasks.invoke(
            {
                "messages": messages,
                "tasks": tasks,
            },
            config,
        )
        return scheduled_tasks

    joiner_prompt=ChatPromptTemplate.from_messages(
        [("system",'''Solve a question answering task. Here are some guidelines:
    - In the Assistant Scratchpad, you will be given results of a plan you have executed to answer the user's question.
    - Thought needs to reason about the question based on the Observations in 1-2 sentences.
    - Ignore irrelevant action results.
    - If the required information is present, give a concise but complete and helpful answer to the user's question.
    - If you are unable to give a satisfactory finishing answer, replan to get the required information. Respond in the following format:
    - The current time is 2105-12-31 23:59:00 in case you need.
    Thought: <reason about the task results and whether you have sufficient information to answer the question>
    Action: <action to take>
    - If an error occurs during previous actions, replan and take corrective measures to obtain the required information.
    - Ensure the final answer is provided in a structured format.
    - The current time is 2105-12-31 23:59:00 in case it is need.
    Available actions:
    (1) Finish(the final answer to return to the user): returns the answer and finishes the task.
    (2) Replan(the reasoning and other information that will help you plan again. Can be a line of any length): instructs why we must replan
    ''' ),
        ("user", '{messages}'),
        ("assistant", '''
        Using the above previous actions, decide whether to replan or finish. 
        If all the required information is present, you may finish. 
        If you have made many attempts to find the information without success, admit so and respond with whatever information you have gathered so the user can work well with you. 
        Ensure to provide the final answer in the following structured JSON format:
        {{'Summary': <concise summary of the answer>,
        'details': <detailed explanation and supporting information>,
        'source': <source of the information or how it was obtained>,
        'final answer':<your final inference as YES or No, True or False, or list of requested information without any extra information>,
        'extra explanation':<put the extra information that you dont provide in final answer here>,
        }}
        In the `'final answer` dont provide additinal explanation or description. Put them in `extra explanation`.
        '''),
        ]
    ).partial(
        examples=""
    )  
    runnable = create_structured_output_runnable(JoinOutputs, llm, joiner_prompt)

    joiner = select_recent_messages | runnable | parse_joiner_output

    graph_builder = MessageGraph()

    # 1.  Define vertices
    # We defined plan_and_schedule above already
    # Assign each node to a state variable to update
    graph_builder.add_node("plan_and_schedule", plan_and_schedule)
    graph_builder.add_node("join", joiner)


    ## Define edges
    graph_builder.add_edge("plan_and_schedule", "join")


    ### This condition determines looping logic

    def should_continue(state: List[BaseMessage]):
        if isinstance(state[-1], AIMessage):
            return "__end__"
        return "plan_and_schedule"


    graph_builder.add_conditional_edges(
        "join",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        #{"plan_and_schedule": "plan_and_schedule", "__end__": "__end__"},
    )
    graph_builder.add_edge(START, "plan_and_schedule")
    chain = graph_builder.compile()

    steps = chain.stream(
        [
            HumanMessage(
                content=question
            )
        ],
        {
            "recursion_limit": 5,
        },
    )
    for step in steps:
        print(step)
        print("---")
    try:
        answer = step['join'][-1].content
        id = step['join'][-1].id
        print(answer, id)
        answer = ast.literal_eval(answer.strip())
        return answer, id
    except:
        return [], 'NA'
            
        
        
def main():
    model="gpt-4o" #gpt-4-turbo-preview
    # Load data from JSON file
    test_file="dataset/mimic_iv_cxr/sampled_test_with_scope_preprocessed_balenced_answer.json"
    m3_lx=[]
    with open(test_file, 'r') as f:
        test_date = json.load(f)
  
    for data in test_date:
        example_question=data['question']
        tables=data['tables']
        print(example_question, tables)
    
        try:
            res, id =run_m3lx(example_question,tables,model)
        except:
            res, id= [], 'NA'
        data['m3lx']=res
        data['smith-langchain-id']=id
        m3_lx.append(data)

    with open('m3lx.json', 'w', encoding='utf-8') as f:
        json.dump(m3_lx, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()