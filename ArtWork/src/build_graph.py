import getpass
import os

from langchain_openai import ChatOpenAI
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



from tools.backend.image_qa import VisualQA
from tools.SQL import get_text2SQL_tools
from tools.visual_qa import get_image_analysis_tools
from tools.plot import get_plotting_tools
from tools.data import get_data_preparation_tools
from langgraph.checkpoint.memory import MemorySaver



def graph_construction(model, temperature, db_path, log_path, saver=None):

    ## Tools
    vqa_model = VisualQA()
    translate= get_text2SQL_tools(ChatOpenAI(model=model, temperature=temperature), db_path)
    image_analysis = get_image_analysis_tools(vqa_model)
    data_preparation= get_data_preparation_tools(ChatOpenAI(model=model, temperature= temperature),log_path=log_path)
    plotting= get_plotting_tools(ChatOpenAI(model=model, temperature= temperature),log_path=log_path)

    tools = [translate,image_analysis,data_preparation,plotting]
    llm = ChatOpenAI(model=model, temperature=temperature)

# In sub_question related to text2SQL include all requested information to be retrieved at once.
 # For each image analysis task, generate three distinct questions that each convey the same idea in different wording.
    prompt = ChatPromptTemplate.from_messages(
      [
        ("system",'''Given a user question, a database schema and tools descriptions, analyze the question to identify and break it down into relevant sub-questions. 
            Determine which tools (e.g., text2SQL, image_analysis, data_preparation, plotting) are appropriate for answering each sub-question based on the available database information and tools.
            Decompose the user question into sub_questions that capture all elements of the question’s intent. This includes identifying the main objective, relevant sub-questions, necessary background information, assumptions, and any secondary requirements. 
            Ensure that no part of the original question’s intent is omitted, and create a list of individual steps to answer the question fully and accurately using tools. 
            You may need to use one tool multiple times to answer to the original question.
            First, you should begin by thoroughly analyzing the user's main question. It’s important to understand the key components and objectives within the query.
            Next, you must review the provided database schema. This involves examining the tables, fields, and relationships within the database to identify which parts of the schema are relevant to the user’s question, and create text2SQL sub-questions.
            For each sub-question, provide all the required information that may required in other tasks. In order to find this information look at the user question and the database inforamtion.
            Ensure you include all necessary information, including columns used for filtering in the retrieve part of the database related task (i.e. Text2SQL), especially when the user question involves plotting or data exploration.
            Each sub-question or step should focus exclusively on a single task.
            In cases where the user’s question involves data that is not directly available in the database schema —such as when there is no corresponding table or column for the required information or image analysis— you must consider the need for image analysis using the image_analysis tools. 
            For instance, if the question involves comparision of image, or asking for specific objects or a object, concepts or a concepts which is not found in the database schema, you must retrieve the `imag_path` and call image analysis task, (e.g, which image depicts <X>).
            The database does not contain any information about what is or are depicted in a painting, to discover that you should include image analysis task.
            This ensures we can address parts of the question that rely on visual data.
            Always include data_preparation task to provide response to user questions in a proper way.
            In case the question asked for plotting, you must include first data_preparation and then the data_plotting tools.
            With a clear understanding of the question and the database schema, you can now break down the main question into smaller, more manageable sub-questions. 
            These sub-questions should each target a specific aspect of the main question. 
            After identifying the sub-questions, you should determine the most appropriate tools to answer each one. Depending on the nature of the sub-questions, we might use a variety of tools.
            Each sub-question should be a textual question. Dont generate a code as a sub-question.
            In any database retreival task, retieve any other columns that may requir for next tasks, especially when the user question involves plotting or data exploration.
            Create a plan to solve it with the utmost parallelizability. 
            Each plan should comprise an action from the following  {num_tools} types:
        {tool_descriptions}
        {num_tools}. 
        join(): Collects and combines results from prior actions.

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
    Thought: <reason about the task results and whether you have sufficient information to answer the question>
    Action: <action to take>
    - If an error occurs during previous actions, replan and take corrective measures to obtain the required information.
    - Ensure that you consider errors in all the previous steps, and tries to replan accordingly.
    - Ensure the final answer is provided in a structured format as JSON as follows:
        {{'Summary': <concise summary of the answer>,
         'details': <detailed explanation and supporting information>,
         'source': <source of the information or how it was obtained>,
         'inference':<your final inference as YES, No, or list of requested information without any extra information which you can take from the `labels` as given below>,
         'extra explanation':<put here the extra information that you dont provide in inference >,
         }}
         In the `inferencer` do not provide additinal explanation or description. Put them in `extra explanation`.

       
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
            return END
        return "plan_and_schedule"

    # Set up memory
    #memory = MemorySaver()  
    graph_builder.add_conditional_edges(
            "join",
            # Next, we pass in the function that will determine which node is called next.
            should_continue,
            #{"plan_and_schedule": "plan_and_schedule", "__end__": "__end__"},
        )
    graph_builder.add_edge(START, "plan_and_schedule")
    chain = graph_builder.compile()
    
    return chain
        

