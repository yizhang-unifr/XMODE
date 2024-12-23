from decimal import Decimal
from uuid import UUID
from datetime import datetime
from tqdm import tqdm as tqdm
from time import sleep
import json
import re
from pathlib import Path
import os
from langsmith import Client

def handle_value(value):
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: handle_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [handle_value(val) for val in value]
    return value

def _extract_run(run):
    if isinstance(run, dict):
        return run
    else:
        keys = run.__dict__.keys()
        temp =  {key: handle_value(run.__dict__[key]) for key in keys}
    if temp['child_runs'] is None:
        temp['child_runs'] = []
    return temp

def get_all_child_runs(root_run, project_name, filter_str = None):
    if not isinstance(root_run, dict):
        root_run = _extract_run(root_run)
    client = Client()
    child_runs = list(client.list_runs(project_name=project_name, run_ids = root_run['child_run_ids'], filter=filter_str, is_root=False))
    child_runs = list(map(_extract_run, child_runs))
    return child_runs

def child2parent_dict(root_run, child_runs):
    if not isinstance(root_run, dict):
        root_run = _extract_run(root_run)
    parent_ids = {root_run['id']: None}
    for child_run in child_runs:
        parent_id = child_run['parent_run_id']
        parent_ids[child_run['id']] = parent_id
    return parent_ids

def get_paths(parent_ids):
    res = {}
    for child_id in parent_ids:
        path = []
        current_id = child_id
        while current_id is not None:
            path.append(current_id)
            current_id = parent_ids[current_id]
        path.reverse()
        res[child_id] = path[:-1]
    return res

def extract_child_runs_by_paths(root_run, project_name):
    root_run = _extract_run(root_run)
    all_child_runs = get_all_child_runs(root_run, project_name)
    parent_ids = child2parent_dict(root_run, all_child_runs)
    all_runs = [root_run] + all_child_runs
    for run in all_runs:
        if parent_ids[run['id']] is not None:
            parent_run = list(filter(lambda x: x['id'] == parent_ids[run['id']], all_runs))[0]
            parent_run['child_runs'].append(run)
    parent_run['child_runs'] = sorted(parent_run['child_runs'], key=lambda x: datetime.strptime(x['start_time'], "%Y-%m-%dT%H:%M:%S.%f"))
    return root_run

def extract_all_child_runs_by_paths(runs, project_name):
    res = []
    for run in tqdm(runs):
        run = extract_child_runs_by_paths(run, project_name)
        res.append(run)
    return res

def sort_child_runs(run):
    if 'child_runs' in run:
        run['child_runs'] = sorted(run['child_runs'], key=lambda x: datetime.strptime(x['start_time'], "%Y-%m-%dT%H:%M:%S.%f"))
        for child_run in run['child_runs']:
            sort_child_runs(child_run)

def extract_and_save_all_child_runs_by_project(project_name, data_path="experiments/xmode/en", filter_str = None):
    client = Client()
    runs = list(client.list_runs(project_name=project_name, filter=filter_str, is_root=True))
    runs = list(map(_extract_run, runs))
    res = extract_all_child_runs_by_paths(runs, project_name=project_name)
    data_path = Path(data_path)
    json_output_path = data_path / f'{project_name}-details.json'
    with open(json_output_path, 'w') as f:
        json.dump(res, f, indent=2)

    # save the item of results in a file folder

    (Path(json_output_path).parent / f"{project_name}-details").mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(res):
        with open(Path(json_output_path).parent / f"{project_name}-details" / f"{i}.json", 'w') as f:
            json.dump(item, f, indent=2)

def check_chain_of_dict(parent_ids):
    # all values should be appears in keys at least once
    all_keys = set(parent_ids.keys())
    all_values = set(parent_ids.values())
    if all_values.issubset(all_keys):
        return []
    else: # find the missing values
        missing_values = all_values - all_keys
        return list(filter(lambda x: x is not None, list(missing_values)))

# if missing values is found, we need to find the missing runs recursively
def find_missing_runs(missing_parent_id, project_name, parent_ids):
    client = Client()
    run = list(client.list_runs(project_name=project_name, run_ids = [missing_parent_id]))[0]
    run = _extract_run(run)
    parent_id = run['parent_run_id']
    # check if parent_id is already in parent_ids
    if parent_id in parent_ids:
        return parent_ids
    else:
        parent_ids[run['id']] = parent_id
        return find_missing_runs(parent_id, parent_ids)
    
    
# extract details

def extract_child_run(data, name):
    if isinstance(data, dict):
        if "name" in data and data["name"] == name:
            return data
        for k, v in data.items():
            res = extract_child_run(v, name)
            if res:
                return res
    elif isinstance(data, list):
        for d in data:
            res = extract_child_run(d, name)
            if res:
                return res
    return None

def extract_child_run_by_id(data, name):
    if isinstance(data, dict):
        if "id" in data and data["id"] == name:
            return data
        for k, v in data.items():
            res = extract_child_run_by_id(v, name)
            if res:
                return res
    elif isinstance(data, list):
        for d in data:
            res = extract_child_run_by_id(d, name)
            if res:
                return res
    return None

# use recursive function to extract the child_runs 
def extract_child_runs(data, name):
    res = []
    if isinstance(data, dict):
        if "name" in data and data["name"] == name:
            res.append(data)
        for k, v in data.items():
            res.extend(extract_child_runs(v, name))
    elif isinstance(data, list):
        for d in data:
            res.extend(extract_child_runs(d, name))
    return res

def extract_parse_joiner_output(run):
    parse_joiner_outputs = extract_child_runs(run, "parse_joiner_output")
    res = []
    for parse_joiner_output in parse_joiner_outputs:
        if parse_joiner_output.get("outputs", False):
            if parse_joiner_output["outputs"].get("output", False):
                if isinstance(parse_joiner_output["outputs"]["output"], list) and len(parse_joiner_output["outputs"]["output"]) > 0:
                    temp = []
                    for output in parse_joiner_output["outputs"]["output"]:
                        temp.append({"content": output["content"]})
                    res.append(temp)
    return res
def get_SQL_query(run):
    res = extract_child_run(run, "PydanticAttrOutputFunctionsParser")
    if res is None:
        res = extract_child_run(run, "PydanticToolsParser")
    if res:
        # output = res['inputs']['input']['additional_kwargs']['function_call']['arguments']
        output = res['outputs']['output']
        if isinstance(output, str):
            output = eval(output)
        return output
    return None

def get_run_by_id(run_id, data):
    if isinstance(data, dict):
        if "id" in data and data["id"] == run_id:
            return data
        for k, v in data.items():
            res = extract_child_run_by_id(v, run_id)
            if res:
                return res
    elif isinstance(data, list):
        for d in data:
            res = extract_child_run_by_id(d, run_id)
            if res:
                return res
    return None

def get_parent_run_ids(run):
    parent_run_ids = run['parent_run_ids']
    if parent_run_ids:
        return [run_id for run_id in parent_run_ids]
    return None

def get_plan_and_schedule_run(xmodeplanparser_run, data):
    parent_runs = get_parent_run_ids(xmodeplanparser_run)
    plan_run = get_run_by_id(parent_runs[-2], data)
    return plan_run

def get_sql_details(schedule_tasks_runs):
    if isinstance(schedule_tasks_runs, dict):
        schedule_tasks_runs = [schedule_tasks_runs]
    text2SQL_runs = []
    for schedul_tasks_run in schedule_tasks_runs:
        all_text2SQL_runs = extract_child_runs(schedul_tasks_run, "text2SQL")
        for all_text2SQL_run in all_text2SQL_runs:
            if all_text2SQL_run.get("outputs", False):
                detail_output = get_SQL_query(all_text2SQL_run)
                if detail_output:
                    output = all_text2SQL_run["outputs"]["output"]
                    # detail_output["output"]["results"] = output
                    # detail_output["input"] = eval(all_text2SQL_run["inputs"]['input'])
                    # text2SQL_runs.append(detail_output)
                    temp = {"data": output, "input": eval(all_text2SQL_run["inputs"]['input']), "output": detail_output}
                    text2SQL_runs.append(temp)
    return text2SQL_runs

def get_image_analysis_details(data):
    if isinstance(data, dict):
        data = [data]
    schedule_tasks_runs = extract_child_runs(data, "schedule_tasks")
    image_analysis_runs = []
    for schedul_tasks_run in schedule_tasks_runs:
        all_image_analysis_runs = extract_child_runs(schedul_tasks_run, "image_analysis")
        for all_image_analysis_run in all_image_analysis_runs:
            if all_image_analysis_run.get("outputs", False):
                detail_output = all_image_analysis_run["outputs"]
                detail_output["input"] = eval(all_image_analysis_run["inputs"]['input'])
                image_analysis_runs.append(detail_output)
        if len(image_analysis_runs) == 0:
            for all_image_analysis_run in all_image_analysis_runs:
                if all_image_analysis_run.get("error", False):
                    detail_output = {}
                    detail_output["error"] = all_image_analysis_run["error"]
                    detail_output["input"] = eval(all_image_analysis_run["inputs"]['input'])
                    detail_output["output"] = all_image_analysis_run.get("outputs", None)
                    image_analysis_runs.append(detail_output)
            
    return image_analysis_runs

def extract_plan_from_M3LXPlanParser(child_run):
    content = child_run['inputs']['input']['content']
    intent_tables = re.findall(r"(\d+).+(intent_tables\(.*\))", content)
    text2sql = re.findall(r"(\d+).+(text2SQL\(.*\))", content)
    image_analysis = re.findall(r"(\d+).+(image_analysis\(.*\))", content)
    join = re.findall(r"(\d+).+(join\(.*\))", content)
    res = intent_tables + text2sql + image_analysis + join
    _result = {}
    for i, f in res:
        if "text2SQL" in f:
            try:
                problem, context = re.findall(r'text2SQL\(problem=["\']?(.+?)["\']?, context=["\']?(\$?.+)["\']?\)', f)[0]
                _result[str(i)] = {
                    "function": f,
                    "problem": problem,
                    "context": context
                }
            except:
                print(f, "error text2SQL")

        elif "intent_table" in f:
            try:
                problem, context = re.findall(r'intent_tables\(problem=["\']?(.+?)["\']?, context=["\']?(\$?.+)["\']?\)', f)[0]
                _result[str(i)] = {
                    "function": f,
                    "problem": problem,
                    "context": context
                }
            except:
                print(f, "error intent_table")

        elif "image_analysis" in f:
            try:
                question, context = re.findall(r'image_analysis\(question=["\']?(.+?)["\']?, context=["\']?\[?(\$?.+)\]?["\']?\)', f)[0]
                _result[str(i)] = {
                    "function": f,
                    "question": question,
                    "context": context
                }
            except:
                print(f, "error image_analysis")
        else:
            _result[str(i)] = {
                "function": f
            }
    return _result

def extract_plan_and_details(data):
    question = data["inputs"]["input"][0]['content'][0]["question"]
    child_runs = extract_child_runs(data, "M3LXPlanParser")
    res = {"plans": []}
    for plan_id, child_run in enumerate(child_runs):
        plan_and_schedule_run = get_plan_and_schedule_run(child_run, data)
        plan = extract_plan_from_M3LXPlanParser(child_run)
        check_image_analysis = 0
        check_text2SQL = 0
        # print(plan)
        text2SQL_runs = get_sql_details(plan_and_schedule_run)
        image_analysis_runs = get_image_analysis_details(plan_and_schedule_run)
        result = {"plan": []}
        for i, p in plan.items():
            if "problem" in p.keys() and 'text2SQL(' in p['function']:
                for text2SQL_run in text2SQL_runs:
                    if p["problem"] == text2SQL_run["input"]["problem"]:
                        _text2SQL_run = text2SQL_run.copy()
                        p["outputs"] = _text2SQL_run
                        check_text2SQL += 1
            elif 'image_analysis(' in p['function']:
                for image_analysis_run in image_analysis_runs:
                    if p["question"] == image_analysis_run["input"]["question"]:
                        _image_analysis_run = image_analysis_run.copy()
                        p["outputs"] = _image_analysis_run
                        check_image_analysis += 1

    # sort the plan by the order of the functions

        for i in sorted([int(key) for key in plan.keys()]):
            plan[str(i)]["id"] = i
            result["plan"].append(plan[str(i)])
        if check_image_analysis == 0:
            result["missing_image_analysis"] = True
        if check_text2SQL == 0:
            result["missing_text2SQL"] = True
        res["plans"].append(result)
    
    predictions = extract_parse_joiner_output(data)
    res["predictions"] = predictions
    res["question"] = question
    return res