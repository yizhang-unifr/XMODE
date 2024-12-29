
import getpass
import os

from langchain_openai import ChatOpenAI

# Imported from the https://github.com/langchain-ai/langgraph/tree/main/examples/plan-and-execute repo
from tools.text2SQL import get_text2SQL_tools
from tools.image_analysis_tool import get_image_analysis_tools
from tools.report_analysis_tool import get_report_analysis_tools
from tools.image_analysis_tool_m3ae import get_image_analysis_tools as m3ae
from tools.intent_table_tool import get_intent_tables_tool


from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from src.joiner import Replan, JoinOutputs
from src.joiner import *


from langchain import hub
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI

import itertools
from src.planner import *
from src.task_fetching_unit import *
from src.joiner import *
from src.joiner import parse_joiner_output
from typing import Dict

from langgraph.graph import END, MessageGraph, START



def graph_construction(model,saver=None):
    # 
    db_path="/home/ubuntu/workspace/XMODE-LLMCompiler/mimic_iv_cxr.db"
    temperature=0
    ## Tools
    translate= get_text2SQL_tools(ChatOpenAI(model=model, temperature=temperature),db_path)
    image_analysis = get_image_analysis_tools(ChatOpenAI(model=model, temperature= temperature),db_path)
    intent_table = get_intent_tables_tool(ChatOpenAI(model=model, temperature=temperature),db_path)
    tools = [intent_table, translate, image_analysis]
    llm = ChatOpenAI(model=model)
   
    prompt = ChatPromptTemplate.from_messages(
    [
    ("system",'''Given a user question and a database schema, analyze the question to identify and break it down into relevant sub-questions. 
     Determine which tools (e.g., text2SQL, image_analysis) are appropriate for answering each sub-question based on the available database information and tools.
     First, you should begin by thoroughly analyzing the user's main question. It’s important to understand the key components and objectives within the query.
     Next, you must review the provided database schema. This involves examining the tables, fields, and relationships within the database to identify which parts of the schema are relevant to the user’s question, and creat a set of sub-questions.
     In cases where the user’s question involves data that is not directly available in the database schema —such as when there is no corresponding table or column for the required information or chest x-ray analysis— you must consider the need for image analysis using the image_analysis tools. 
     For instance, if the question involves comparision of studies, images, chest x-rays  for specific abnormality, disease or finding in the paitent study and the relevant column (e.g., abnormality, findings, chest x-ray) is not found in the database schema, you must retrieve th`image_id` of the study for image analysis, 
     This ensures we can address parts of the question that rely on visual data. If the question includes the study ID, you can directly initiate the image analysis task using the mentioned id for a study.
     With a clear understanding of the question and the database schema, you can now break down the main question into smaller, more manageable sub-questions. 
     These sub-questions should each target a specific aspect of the main question. 
     After identifying the sub-questions, you should determine the most appropriate tools to answer each one. Depending on the nature of the sub-questions, we might use a variety of tools.
     Each sub-question should be a textual question. Dont generate a code as a sub-question.
     Each image analysis task should consider only one study. Study id is different than image id. 
     Include the database schema in the context for text2SQL tasks. 
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
    planner = create_planner(llm, tools, prompt)
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
    - Ensure that you consider errors in all the previous steps, and tries to replan accordingly.
    - The current time is 2105-12-31 23:59:00 in case it is need.
    - Ensure the final answer is provided in a structured format as JSON as follows:
        {{'Summary': <concise summary of the answer>,
         'details': <detailed explanation and supporting information>,
         'source': <source of the information or how it was obtained>,
         'inference':<your final inference as YES, No, or list of requested information without any extra information which you can take from the `labels` as given below>,
         'extra explanation':<put here the extra information that you dont provide in inference >,
         }}
         In the `inferencer` do not provide additinal explanation or description. Put them in `extra explanation`.
         You should provide the `inference` in the final answer as JSON format by choosing the possible labels from the follwing list:
         "no",
         "lung lesion",
         "pleural/parenchymal scarring",
         "left apical zone",
         "neck",
         "interstitial lung disease",
         "pulmonary edema/hazy opacity",
         "left lower lung zone",
         "cyst/bullae",
         "superior mediastinal mass/enlargement",
         "linear/patchy atelectasis",
         "hernia",
         "atelectasis",
         "endotracheal tube",
         "prosthetic valve",
         "right mid lung zone",
         "subcutaneous air",
         "pericardial effusion",
         "pneumothorax",
         "left lung",
         "breast/nipple shadows",
         "right shoulder",
         "fluid overload/heart failure",
         "granulomatous disease",
         "left hilar structures",
         "mediastinal displacement",
         "rib fracture",
         "right lower lung zone",
         "aortic arch",
         "cardiac pacer and wires",
         "aspiration",
         "airspace opacity",
         "intra-aortic balloon pump",
         "clavicle fracture",
         "vascular congestion",
         "subclavian line",
         "enlarged cardiac silhouette",
         "spine",
         "hydropneumothorax",
         "hyperaeration",
         "pleural effusion",
         "vascular redistribution",
         "right apical zone",
         "tracheostomy tube",
         "right lung",
         "shoulder osteoarthritis",
         "m",
         "pa",
         "right upper lung zone",
         "right chest wall",
         "goiter",
         "low lung volumes",
         "lung cancer",
         "tortuous aorta",
         "upper mediastinum",
         "copd/emphysema",
         "right costophrenic angle",
         "left clavicle",
         "spinal fracture",
         "bone lesion",
         "alveolar hemorrhage",
         "left breast",
         "ij line",
         "pneumonia",
         "lobar/segmental collapse",
         "left mid lung zone",
         "infiltration",
         "left hemidiaphragm",
         "chest tube",
         "svc",
         "left upper lung zone",
         "left costophrenic angle",
         "enlarged hilum",
         "f",
         "carina",
         "trachea",
         "right hilar structures",
         "scoliosis",
         "mediastinum",
         "left chest wall",
         "mediastinal drain",
         "rotated",
         "consolidation",
         "cardiac silhouette",
         "right atrium",
         "pneumomediastinum",
         "right clavicle",
         "yes",
         "picc",
         "swan-ganz catheter",
         "multiple masses/nodules",
         "ap",
         "vascular calcification",
         "right hemidiaphragm",
         "spinal degenerative changes",
         "abdomen",
         "costophrenic angle blunting",
         "lung opacity",
         "calcified nodule",
         "cabg grafts",
         "chest port",
         "enteric tube",
         "left shoulder",
         "right breast",
         "elevated hemidiaphragm",
         "cavoatrial junction",
         "increased reticular markings/ild pattern",
         "mediastinal widening",
         "mass/nodule (not otherwise specified)",
         "sub-diaphragmatic air"
         
    Available actions:
    (1) Finish(the final answer to return to the user): returns the answer and finishes the task.
    (2) Replan(the reasoning and other information that will help you plan again. Can be a line of any length): instructs why we must replan
    ''' ),
        ("user", '{messages}'),
        ("assistant", '''
        Using the above previous actions, decide whether to replan or finish. 
        If all the required information is present, you may finish. 
        If you have made many attempts to find the information without success, admit so and respond with whatever information you have gathered so the user can work well with you. 
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
    chain = graph_builder.compile(checkpointer=saver)
    # _steps_dict={}
    # for idx, step in enumerate(steps):
    #     _step_dict={}
    #     for k,v in step.items():
    #         _step_dict[k]=[item.__dict__ for item in v]
    #     _steps_dict[idx]=_step_dict
    
    
    return chain

def graph_construction_m3ae(model,saver=None):
    # 
    db_path="/home/ubuntu/workspace/XMODE-LLMCompiler/mimic_iv_cxr.db"
    temperature=0
    ## Tools
    translate= get_text2SQL_tools(ChatOpenAI(model=model, temperature=temperature),db_path)
    image_analysis = m3ae(db_path)
    intent_tables = get_intent_tables_tool(ChatOpenAI(model=model, temperature=temperature),db_path)
    tools = [intent_tables, translate, image_analysis]
    llm = ChatOpenAI(model=model,temperature=temperature)
   
    prompt = ChatPromptTemplate.from_messages(
    [
    ("system",'''Given a user question, a database schema and tools descriptions, analyze the question to identify and break it down into relevant sub-questions. 
     Determine which tools (e.g., intent_tables, text2SQL, image_analysis) are appropriate for intenting tables or answering each sub-question based on the available database information and tools.
     Decompose the user question into sub_questions that capture all elements of the question’s intent. This includes identifying the main objective, relevant sub-questions, necessary background information, assumptions, and any secondary requirements. 
     Ensure that no part of the original question’s intent is omitted, and create a list of individual steps to answer the question fully and accurately using tools. 
     You may need to use one tool multiple times to answer to the original question.
     First, you should begin by thoroughly analyzing the user's main question and intent tables related to the question. It’s important to understand the key components and objectives within the query.
     Next, you must review the provided database schema. This involves examining the tables, fields, and relationships within the database to identify which parts of the schema are relevant to the user’s question using only one intent_tables tool, and create text2SQL sub-questions.
     For each sub-question, provide all the required information that may required in other tasks. In order to find this information look at the user question and the database inforamtion.
     Ensure you include all necessary information, including columns used for filtering in the retrieve part of the database related task (i.e. text2SQL), especially when the user question involves questions about x-ray and other tables in database.
     Each sub-question or step should focus exclusively on a single task.
     In cases where the user’s question involves data that is not directly available in the database schema —such as when there is no corresponding table or column for the required information or chest x-ray analysis— you must consider the need for image analysis using the image_analysis tools. 
     For instance, if the question involves comparision of studies, images, chest x-rays  for specific abnormality, disease or finding in the paitent study and the relevant column (e.g., abnormality, findings, chest x-ray) is not found in the database schema, you must retrieve th`image_id` of the study for image analysis, 
     This ensures we can address parts of the question that rely on visual data. If the question includes the study ID, you can directly initiate the image analysis task using the mentioned id for a study.
     With a clear understanding of the question and the database schema, you can now break down the main question into smaller, more manageable sub-questions. 
     These sub-questions should each target a specific aspect of the main question. 
     After identifying the sub-questions, you should determine the most appropriate tools to answer each one. Depending on the nature of the sub-questions, we might use a variety of tools.
     Each sub-question should be a textual question. Dont generate a code as a sub-question.
     Each image analysis task should consider only one study. Study id is different than image id. 
     Include the intented database schema in the context for text2SQL tasks. 
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
    planner = create_planner(llm, tools, prompt)
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
    - Please ALWAYS keep in mind that the intented tables MUST be identified a prior the text2SQL tasks.
    - Thought needs to reason about the question based on the Observations in 1-2 sentences.
    - Ignore irrelevant action results.
    - If the required information is present, give a concise but complete and helpful answer to the user's question.
    - If you are unable to give a satisfactory finishing answer, replan to get the required information. Respond in the following format:
    - The current time is 2105-12-31 23:59:00 in case you need.
    Thought: <reason about the task results and whether you have sufficient information to answer the question>
    Action: <action to take>
    - If an error occurs during previous actions, replan and take corrective measures to obtain the required information.
    - Ensure that you consider errors in all the previous steps, and tries to replan accordingly.
    - The current time is 2105-12-31 23:59:00 in case it is need.
    - Ensure the final answer is provided in a structured format as JSON as follows:
        {{'Summary': <concise summary of the answer>,
         'details': <detailed explanation and supporting information>,
         'source': <source of the information or how it was obtained>,
         'inference':<your final inference as YES, No, or list of requested information without any extra information which you can take from the `labels` as given below>,
         'extra explanation':<put here the extra information that you dont provide in inference >,
         }}
         In the `inferencer` do not provide additinal explanation or description. Put them in `extra explanation`.
         You should provide the `inference` in the final answer as JSON format by choosing the possible labels from the follwing list:
         "no",
         "lung lesion",
         "pleural/parenchymal scarring",
         "left apical zone",
         "neck",
         "interstitial lung disease",
         "pulmonary edema/hazy opacity",
         "left lower lung zone",
         "cyst/bullae",
         "superior mediastinal mass/enlargement",
         "linear/patchy atelectasis",
         "hernia",
         "atelectasis",
         "endotracheal tube",
         "prosthetic valve",
         "right mid lung zone",
         "subcutaneous air",
         "pericardial effusion",
         "pneumothorax",
         "left lung",
         "breast/nipple shadows",
         "right shoulder",
         "fluid overload/heart failure",
         "granulomatous disease",
         "left hilar structures",
         "mediastinal displacement",
         "rib fracture",
         "right lower lung zone",
         "aortic arch",
         "cardiac pacer and wires",
         "aspiration",
         "airspace opacity",
         "intra-aortic balloon pump",
         "clavicle fracture",
         "vascular congestion",
         "subclavian line",
         "enlarged cardiac silhouette",
         "spine",
         "hydropneumothorax",
         "hyperaeration",
         "pleural effusion",
         "vascular redistribution",
         "right apical zone",
         "tracheostomy tube",
         "right lung",
         "shoulder osteoarthritis",
         "m",
         "pa",
         "right upper lung zone",
         "right chest wall",
         "goiter",
         "low lung volumes",
         "lung cancer",
         "tortuous aorta",
         "upper mediastinum",
         "copd/emphysema",
         "right costophrenic angle",
         "left clavicle",
         "spinal fracture",
         "bone lesion",
         "alveolar hemorrhage",
         "left breast",
         "ij line",
         "pneumonia",
         "lobar/segmental collapse",
         "left mid lung zone",
         "infiltration",
         "left hemidiaphragm",
         "chest tube",
         "svc",
         "left upper lung zone",
         "left costophrenic angle",
         "enlarged hilum",
         "f",
         "carina",
         "trachea",
         "right hilar structures",
         "scoliosis",
         "mediastinum",
         "left chest wall",
         "mediastinal drain",
         "rotated",
         "consolidation",
         "cardiac silhouette",
         "right atrium",
         "pneumomediastinum",
         "right clavicle",
         "yes",
         "picc",
         "swan-ganz catheter",
         "multiple masses/nodules",
         "ap",
         "vascular calcification",
         "right hemidiaphragm",
         "spinal degenerative changes",
         "abdomen",
         "costophrenic angle blunting",
         "lung opacity",
         "calcified nodule",
         "cabg grafts",
         "chest port",
         "enteric tube",
         "left shoulder",
         "right breast",
         "elevated hemidiaphragm",
         "cavoatrial junction",
         "increased reticular markings/ild pattern",
         "mediastinal widening",
         "mass/nodule (not otherwise specified)",
         "sub-diaphragmatic air"

    Available actions:
    (1) Finish(the final answer to return to the user): returns the answer and finishes the task.
    (2) Replan(the reasoning and other information that will help you plan again. Can be a line of any length): instructs why we must replan
    ''' ),
        ("user", '{messages}'),
        ("assistant", '''
        Using the above previous actions, decide whether to replan or finish. 
        If all the required information is present, you may finish. 
        If you have made many attempts to find the information without success, admit so and respond with whatever information you have gathered so the user can work well with you.      
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
    # _steps_dict={}
    # for idx, step in enumerate(steps):
    #     _step_dict={}
    #     for k,v in step.items():
    #         _step_dict[k]=[item.__dict__ for item in v]
    #     _steps_dict[idx]=_step_dict
    
    
    return chain

def graph_construction_report(model,saver=None):
    # 
    db_path="/home/ubuntu/workspace/XMODE-LLMCompiler/mimic_iv_cxr.db"
    temperature=0
    ## Tools
    translate= get_text2SQL_tools(ChatOpenAI(model=model, temperature=temperature),db_path)
    report_analysis = get_report_analysis_tools(ChatOpenAI(model=model, temperature= temperature),db_path)
    tools = [translate,report_analysis]
    llm = ChatOpenAI(model=model)
   
    prompt = ChatPromptTemplate.from_messages(
    [
    ("system",'''Given a user question and a database schema, analyze the question to identify and break it down into relevant sub-questions. 
     Determine which tools (e.g., text2SQL, reoort_analysis) are appropriate for answering each sub-question based on the available database information and tools.
     First, you should begin by thoroughly analyzing the user's main question. It’s important to understand the key components and objectives within the query.
     Next, you must review the provided database schema. This involves examining the tables, fields, and relationships within the database to identify which parts of the schema are relevant to the user’s question, and creat a set of sub-questions.
     In cases where the user’s question involves data that is not directly available in the database schema —such as when there is no corresponding table or column for the required information or chest x-ray analysis— you must consider the need for report analysis using the report_analysis tools. 
     For instance, if the question involves comparision of studies, images, chest x-rays  for specific abnormality, disease or finding in the paitent study and the relevant column (e.g., abnormality, findings, chest x-ray) is not found in the database schema, you must retrieve th `study_id` of the study for report analysis, 
     this ensures we can address parts of the question that rely on textual data. If the question includes the study ID, you can directly initiate the report analysis task.
     With a clear understanding of the question and the database schema, you can now break down the main question into smaller, more manageable sub-questions. 
     These sub-questions should each target a specific aspect of the main question. 
     After identifying the sub-questions, you should determine the most appropriate tools to answer each one. Depending on the nature of the sub-questions, we might use a variety of tools.
     Each sub-question should be a textual question. Dont generate a code as a sub-question.
     Each report analysis task should consider only one study.
     Include the database schema in the context for text2SQL tasks. 
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
    planner = create_planner(llm, tools, prompt)
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
    - Ensure that you consider errors in all the previous steps, and tries to replan accordingly.
    - The current time is 2105-12-31 23:59:00 in case it is need.
    - Ensure the final answer is provided in a structured format as JSON as follows:
        {{'Summary': <concise summary of the answer>,
         'details': <detailed explanation and supporting information>,
         'source': <source of the information or how it was obtained>,
         'inference':<your final inference as YES, No, or list of requested information without any extra information which you can take from the `labels` as given below>,
         'extra explanation':<put here the extra information that you dont provide in inference >,
         }}
         In the `inferencer` do not provide additinal explanation or description. Put them in `extra explanation`.
         You should provide the `inference` in the final answer as JSON format by choosing the possible labels from the follwing list:
         "no",
         "lung lesion",
         "pleural/parenchymal scarring",
         "left apical zone",
         "neck",
         "interstitial lung disease",
         "pulmonary edema/hazy opacity",
         "left lower lung zone",
         "cyst/bullae",
         "superior mediastinal mass/enlargement",
         "linear/patchy atelectasis",
         "hernia",
         "atelectasis",
         "endotracheal tube",
         "prosthetic valve",
         "right mid lung zone",
         "subcutaneous air",
         "pericardial effusion",
         "pneumothorax",
         "left lung",
         "breast/nipple shadows",
         "right shoulder",
         "fluid overload/heart failure",
         "granulomatous disease",
         "left hilar structures",
         "mediastinal displacement",
         "rib fracture",
         "right lower lung zone",
         "aortic arch",
         "cardiac pacer and wires",
         "aspiration",
         "airspace opacity",
         "intra-aortic balloon pump",
         "clavicle fracture",
         "vascular congestion",
         "subclavian line",
         "enlarged cardiac silhouette",
         "spine",
         "hydropneumothorax",
         "hyperaeration",
         "pleural effusion",
         "vascular redistribution",
         "right apical zone",
         "tracheostomy tube",
         "right lung",
         "shoulder osteoarthritis",
         "m",
         "pa",
         "right upper lung zone",
         "right chest wall",
         "goiter",
         "low lung volumes",
         "lung cancer",
         "tortuous aorta",
         "upper mediastinum",
         "copd/emphysema",
         "right costophrenic angle",
         "left clavicle",
         "spinal fracture",
         "bone lesion",
         "alveolar hemorrhage",
         "left breast",
         "ij line",
         "pneumonia",
         "lobar/segmental collapse",
         "left mid lung zone",
         "infiltration",
         "left hemidiaphragm",
         "chest tube",
         "svc",
         "left upper lung zone",
         "left costophrenic angle",
         "enlarged hilum",
         "f",
         "carina",
         "trachea",
         "right hilar structures",
         "scoliosis",
         "mediastinum",
         "left chest wall",
         "mediastinal drain",
         "rotated",
         "consolidation",
         "cardiac silhouette",
         "right atrium",
         "pneumomediastinum",
         "right clavicle",
         "yes",
         "picc",
         "swan-ganz catheter",
         "multiple masses/nodules",
         "ap",
         "vascular calcification",
         "right hemidiaphragm",
         "spinal degenerative changes",
         "abdomen",
         "costophrenic angle blunting",
         "lung opacity",
         "calcified nodule",
         "cabg grafts",
         "chest port",
         "enteric tube",
         "left shoulder",
         "right breast",
         "elevated hemidiaphragm",
         "cavoatrial junction",
         "increased reticular markings/ild pattern",
         "mediastinal widening",
         "mass/nodule (not otherwise specified)",
         "sub-diaphragmatic air"
         
    Available actions:
    (1) Finish(the final answer to return to the user): returns the answer and finishes the task.
    (2) Replan(the reasoning and other information that will help you plan again. Can be a line of any length): instructs why we must replan
    ''' ),
        ("user", '{messages}'),
        ("assistant", '''Using the above previous actions, decide whether to replan or finish. 
         If all the required information is present, you may finish. 
         If you have made many attempts to find the information without success, admit so and respond with whatever information you have gathered so the user can work well with you. 
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
    chain = graph_builder.compile(checkpointer=saver)
    # _steps_dict={}
    # for idx, step in enumerate(steps):
    #     _step_dict={}
    #     for k,v in step.items():
    #         _step_dict[k]=[item.__dict__ for item in v]
    #     _steps_dict[idx]=_step_dict
    
    
    return chain