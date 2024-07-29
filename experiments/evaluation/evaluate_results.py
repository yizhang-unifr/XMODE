import json
from pathlib import Path
import sys
import re
import ast
import argparse
import pandas as pd
from datetime import datetime

# adapt the path to the root directory
root_path = Path(__file__).resolve().parents[2]

# use argparse to get the path to the file

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
    try:
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
    except Exception as e:
        if isinstance(source, dict):
            return source
        else:
            print(f"Error in parsing the source: {source}, {type(source)}")
            return source
        

def extract_prediction(_data: list):
    # only get the first prediction
    # Check if it has the following keys
    if len(_data) == 0:
        return {
            "pred_summary": None,
            "pred_details": None,
            "pred_source": None,
            "predict_answer": None,
            "pred_explanation": None
        }
    if isinstance(_data, str):
        try:
            _data = ast.literal_eval(format_text_for_malformed_prediction(_data))
            if not isinstance(_data, list):
                _data = [_data]
        except Exception as e:
            print(f"Error in parsing the data: {_data}")
            return {
                "pred_summary": None,
                "pred_details": None,
                "pred_source": None,
                "predict_answer": _data,
                "pred_explanation": None
            }
    data = _data[0]
    keys = {"Summary":"pred_summary", "details": "pred_details", "source": "pred_source", "inference": "predict_answer", "extra explanation": "pred_explanation"}
    res = {}
    for key in keys:
        if key == 'source':
            temp = parse_source(data.get(key, None))
        assert isinstance(data, dict), f"Expected a dictionary, but got {type(data)} {_data}"
        temp = data.get(key, None)
        res[keys[key]] = temp
    return res

def analyze_datasets(data_path, output_path = "output", evaluation_path="evaluation"):
    timestamp = get_current_datetime()
    file_df = extract_files(data_path)
    summary_list = []
    output_path = output_path + "/" + timestamp
    evaluation_path = evaluation_path + "/" + timestamp
    for index, row in file_df.iterrows():
        lang = row["lang"]
        data = load_json(row["path"])
        res = []
        for i, item in enumerate(data):
            temp = {
                    "db_id": item['db_id'],
                    "scope": item['scope'],
                    "question": item['question'],
                    "answer": item['answer'],
                }
            if lang in ["zh", "de"]:
                if lang == "zh":
                    _lang = "zh_cn"
                else:
                    _lang = lang
                temp.update({
                    f"question_{lang}": item.get(f'question_{_lang}', item['question']),
                    f"answer_{lang}": item.get(f'answer_{_lang}', item['answer'])})
            if item.get('prediction', None):
                temp.update(extract_prediction(item['prediction']))
            else:
                temp.update({
                    "pred_summary": None,
                    "pred_details": None,
                    "pred_source": None,
                    "predict_answer": None,
                    "pred_explanation": None
                })
            res.append(temp)
        output_df = pd.DataFrame(res)
        eval_df, accuracy = analyze_output(output_df)
        
        # save the output information
        _output_path = Path(row["path"]).parents[1] / output_path / row["lang"]
        _output_path.mkdir(parents=True, exist_ok=True)
        output_file_name = row["experiment"] + "_output.csv"
        output_file_path = _output_path / output_file_name
        output_df.to_csv(output_file_path, index=False)
        if output_file_path.exists():
            print(f"Saved the output file to {output_file_path}")
        else:
            raise ValueError(f"Could not save the file to {output_file_path}")
        row["output_file"] = output_file_path.as_posix()
        
        # save the evaluation information
        _evaluation_path = Path(row["path"]).parents[1] / evaluation_path / row["lang"]
        _evaluation_path.mkdir(parents=True, exist_ok=True)
        evaluation_file_name = row["experiment"] + "_evaluation.csv"
        evaluation_file_path = _evaluation_path / evaluation_file_name
        eval_df.to_csv(evaluation_file_path, index=False)
        if evaluation_file_path.exists():
            print(f"Saved the file to {evaluation_file_path}")
        else:
            raise ValueError(f"Could not save the evaluation file to {evaluation_file_path}")
        row["evaluation_file"] = evaluation_file_path.as_posix()
        row["gen_acc"] = accuracy
        summary_list.append(row)
    new_file_df = pd.DataFrame(summary_list)
    
    new_file_df.to_csv(f"{data_path}/evaluation_summary_{timestamp}.csv", index=False)
    return new_file_df
        
def get_current_datetime():
    timestemp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return timestemp

def get_binary_label(x, y):
    y = y[0]
    res = -1
    not_list = ['no', 'not']
    yes_list = ['yes', 'detected']
    if x == None:
        return res == y
    for no in not_list:
        if no in x.lower():
            res = 0
            break
    for yes in yes_list:
        if yes in x.lower():
            if res == 0:
                res = -1
                return res == y
            else:
                res = 1
                break
    return res == y

def get_list_results(x, y):
    if len(x) != len(y):
        return False
    # check if all elements in x is in y
    for i in x:
        if i not in y:
            return False
    # check if all elements in y is in x
    for i in y:
        if i not in x:
            return False
    return True

def correct_malformed_json(context):
    context = re.sub(r"\'", r'"', context)
    context = re.sub(r"None", r'null', context)
    context = re.sub(r"True", r'true', context)
    context = re.sub(r"False", r'false', context)
    context = re.sub(r"nan", r'"nan"', context)
    context = re.sub(r"-inf", r'"-inf"', context)
    context = re.sub(r"\"{", r"{", context)
    context = re.sub(r"}\"", r"}", context)
    context = context.replace("\n", " ")
    return context

def format_text_for_malformed_prediction(text):
    # Replacement patterns and their corresponding replacements
    replacements = [
        (r"{\s*'", '{"'),
        (r"'\s*}", '"}'),
        (r"'\s*:\s*\[\s*'", '": ["'),
        (r"'\s*\]\s*,\s*'", '"], "'),
        (r"\[\s*'", '["'),
        (r"'\s*\]", '"]'),
        (r"'\s*:\s*'", '": "'),
        (r"\s*',\s*'", '", "'),
        (r"\n", " "),
        (r"}+", "}")
    ]

    # Applying all replacements atomically
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)

    return text

def parse_str(x):
    try:
        return ast.literal_eval(x)
    except SyntaxError as e:
        return ast.literal_eval(correct_malformed_json(x))
    except Exception as e:
        print(f"Error in parsing:\n{x}: {e}")
    finally:
        return x

def get_results(x, y):
    if isinstance(x, str):
        x = parse_str(x)
    if isinstance(y, str):
        y = parse_str(y)
    if len(y) == 1 and isinstance(y[0], int) and isinstance(x, str):
        return int(get_binary_label(x, y))
    if isinstance(x, list) and isinstance(y, list):
        return int(get_list_results(x, y))
    else:
        return -1

def analyze_output(output_df):
    eval_df = output_df.copy()
    labels = eval_df.apply(lambda x: get_results(x["predict_answer"], x["answer"]), axis=1)
    eval_df["label"] = labels
    accuracy = labels[labels == 1].count() / len(labels)
    return eval_df, accuracy
        
    

if __name__ == "__main__":
    print(root_path)
    print(extract_files(args.data_path))
    analyze_datasets(args.data_path)