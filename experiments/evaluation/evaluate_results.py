import json
from pathlib import Path
import sys
import re
import ast
import argparse
import pandas as pd

# adapt the path to the root directory
root_path = Path(__file__).resolve().parents[2]

# use argparse to get the path to the file
"""
def correct_malformed_json(context):
    context = re.sub(r"(\w+):", r'"\1":', context)
    context = re.sub(r"\'", r'"', context)
    context = re.sub(r"None", r'null', context)
    context = re.sub(r"True", r'true', context)
    context = re.sub(r"False", r'false', context)
    context = re.sub(r"nan", r'"nan"', context)
    context = re.sub(r"inf", r'"inf"', context)
    context = re.sub(r"-inf", r'"-inf"', context)
    context = re.sub(r"\'", r'"', context)
    context = re.sub(r"\"\"", r'"', context)
    context = re.sub(r"\"{", r"{", context)
    context = re.sub(r"}\"", r"}",
"""

def load_args():                 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=False, default=f"{root_path}/experiments/m3lx/")
    
    return parser.parse_args()

args = load_args()

def get_type(file_name):
    qa_type = None
    model_type = None
    valid_qa_types = ["vqa", "qa"] # order matters!!!
    valid_model_types = ["openai", "m3ae"] 
    for qtype in valid_qa_types:
        if qtype in file_name:
            qa_type = qtype
            break
    for mtype in valid_model_types:
        if mtype in file_name:
            model_type = mtype
            break
    if qa_type and model_type:
        return qa_type, model_type
    raise ValueError(f"Could not determine the type of the file: {file_name}")

def extract_files(data_path):
    # load all files in the sub directory
    languages = ["de", "en", "zh"]
    file_list = []
    for lang in languages:
        sub_path = Path(f"{data_path}/{lang}", recursive=False)
        sub_files = sub_path.glob("*.json")
        sub_files = [f for f in sub_files if Path(f).is_file()]
        for sf in sub_files:
            file_name = sf.stem
            qa_type, model_type = get_type(file_name)
            d = {
                "lang": lang,
                "qa_type": qa_type,
                "model_type": model_type,
                "path": sf.as_posix(),
                "experiment": file_name
            }
            file_list.append(d)
    file_df = pd.DataFrame(file_list)
    return file_df

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def parse_source(source: str):
    """input
    Image analysis of last study (image_id: fc252cec-cb2bd316-a6df094b-dfd88ffa-b9315b50) and previous study (image_id: 13ba579e-736450c7-f8722c82-7cb94950-f21fe768)
    Output:
    {
        "image_id": ["fc252cec-cb2bd316-a6df094b-dfd88ffa-b9315b50", "13ba579e-736450c7-f8722c82-7cb94950-f21fe768"]
    }
    """
    image_id_pattern = re.compile(r"([a-f0-9-]{44})")
    image_id_res = image_id_pattern.findall(source)
    # pattern for getting  8-digits number
    study_id_pattern = re.compile(r"([0-9]{8})")
    
    study_id_res = study_id_pattern.findall(source)
    res = {
        "image_id": image_id_res,
        "study_id": study_id_res,
        "text": source
    }
    return res

def extract_prediction(data: list):
    # only get the first prediction
    # Check if it has the following keys
    if len(data) == 0:
        return {
            "pred_summary": None,
            "pred_details": None,
            "pred_source": None,
            "predict_answer": None,
            "pred_explanation": None
        }
    data = data[0]
    keys = {"Summary":"pred_summary", "details": "pred_details", "source": "pred_source", "final answer": "predict_answer", "extra explanation": "pred_explanation"}
    res = {}
    for key in keys:
        if key == 'source':
            temp = parse_source(data.get(key, None))
        temp = data.get(key, None)
        res[keys[key]] = temp
    return res

def extract_information(data_path):
    file_df = extract_files(data_path)
    for index, row in file_df.iterrows():
        data = load_json(row["path"])
        res = []
        for i, item in enumerate(data):
            if item.get('prediction', None):
                temp = {
                    "db_id": item['db_id'],
                    "question": item['question'],
                    "answer": item['answer'],
                }
            temp.update(extract_prediction(item['prediction']))
            res.append(temp)
        print(data)
        break
    

if __name__ == "__main__":
    print(root_path)
    extract_information(args.data_path)