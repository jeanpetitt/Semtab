from flask import Flask, request, jsonify
from task.ra import RATask
from task.cpa import CPATask
from task.cea import CEATask
from task.cta import CTATask
from task.td import TDTask
from evaluation.cta_evaluator import CTA_Evaluator
from evaluation.ra_evaluator import RA_Evaluator
from evaluation.td_evaluator import TD_Evaluator
from evaluation.cpa_evaluator import CPA_Evaluator
import os
from path_data.utils import ra_model_finetuned, cpa_model_finetuned, cea_model_finetuned_4, td_model_finetuned_2, td_model_finetuned_1

app = Flask(__name__)

dataset = []
# Route to generate  dataset base on tabular datase 
@app.route('/make_dataset', methods=['POST'])
def make_dataset():
    # Fill form to make dataset
    task = request.json.get('task', "cea")
    dataset_name = request.json.get('dataset_name', "tsotsa")
    target_path = request.json.get('target_path', "")
    target_gt_path = request.json.get('target_gt_path', "")
    table_path = request.json.get('table_path', "")
    is_train = request.json.get('is_train', True)
    header = request.json.get('header', True)
    path = f"results/dataset/{task}"
    # End Form
    if task.lower() == "ra":
        ra_task = RATask(
            dataset_name=dataset_name,
            table_path=table_path,
            target_file=target_path, 
            target_file_gt=target_gt_path
        )
        if not os.path.exists(path):
            path = os.makedirs(path)
        if is_train:
            ra_task.set_dataset_path(f"{path}/dataset_{ra_task.get_dataset_name()}_{task}_train.csv")
        else:
            ra_task.set_dataset_path(f"{path}/dataset_{ra_task.get_dataset_name()}_{task}_test.csv")
        try:
            ra_task._makeDataset(header=header, is_train=is_train)
            return jsonify({
                "dataset_path": ra_task.get_dataset_path(),
                "table_path": ra_task.target_file_gt,
                "task": "Row Annotation",
                "status": "succes",
                "code": 200
            })
        except Exception as e:
            print(e)
            return jsonify({
                "message": "Error during the process:",
            }), 400
    elif task.lower() == 'cpa':
        is_entity = request.json.get('is_entity', True)
        is_horizontal = request.json.get('is_horizontal', False)
        cpa_task = CPATask(
            dataset_name=dataset_name,
            table_path=table_path,
            target_file=target_path, 
            target_file_gt=target_gt_path
        )
        if not os.path.exists(path):
            path = os.makedirs(path)
        if is_train:
            cpa_task.set_dataset_path(f"{path}/dataset_{cpa_task.get_dataset_name()}_{task}_train.csv")
        else:
            cpa_task.set_dataset_path(f"{path}/dataset_{cpa_task.get_dataset_name()}_{task}_test.csv")
        cpa_task._makeDataset(header=header, is_train=is_train, is_horizontal=is_horizontal, is_entity=is_entity)
        try:
            return jsonify({
                "dataset_path": cpa_task.get_dataset_path(),
                "table_path": cpa_task.target_file_gt,
                "task": "Column Property Annotation",
                "status": "succes",
                "code": 200
            })
        except Exception as e:
            print(e)
            return jsonify({
                "message": "Error during the process:",
            }), 400
    elif task.lower() == "cea":       
        is_vertical = request.json.get('is_vertical', False)
        transpose = request.json.get('transpose', False)
        col_before_row = request.json.get('col_before_row', False)
        coma_in_cell = request.json.get('coma_in_cell', False)
        split = request.json.get('split', False)
        cea_task = CEATask(
            dataset_name=dataset_name,
            table_path=table_path,
            target_file=target_path, 
            target_file_gt=target_gt_path
        )
        if not os.path.exists(path):
            path = os.makedirs(path)
        if is_train:
            cea_task.set_dataset_path(f"{path}/dataset_{cea_task.get_dataset_name()}_{task}_train.csv")
        else:
            cea_task.set_dataset_path(f"{path}/dataset_{cea_task.get_dataset_name()}_{task}_test.csv")
        cea_task._makeDataset(
            header=header, 
            is_vertical=is_vertical, 
            transpose=transpose, 
            col_before_row=col_before_row,
            comma_in_cell=coma_in_cell,
            is_train=is_train,
            split=split
        )
        try:
            return jsonify({
                "dataset_path": cea_task.get_dataset_path(),
                "table_path": cea_task.target_file_gt,
                "task": "Column Property Annotation",
                "status": "succes",
                "code": 200
            })
        except Exception as e:
            print(e)
            return jsonify({
                "message": "Error during the process:",
            }), 400
    elif task.lower() == "cta":
        return jsonify({"message": task}), 200
    elif task.lower() == "td":
        return jsonify({"message": task}), 200
    else:
        return jsonify({"message": "task does not exist"}), 400
    

# Route to annote tabular data
@app.route('/annotate', methods=['POST'])
def annotate():
    # Fill form to make dataset
    task = request.json.get('task', "")
    dataset_name = request.json.get('dataset_name', "tsotsa")
    dataset_path = request.json.get('dataset_path', "")
    is_train = request.json.get('table_path', True)
    split = request.json.get('split', 0)
    is_entity = request.json.get('is_entity', False)
    path = f"results/annotate/{task}"
    # End Form
    if task.lower() == "ra":
        ra_task = RATask(
            dataset_name=dataset_name,
            output_dataset=dataset_path
        )
        if not os.path.exists(path):
            path = os.makedirs(path)
        if is_train:
            ra_task.set_annotated_file_path(f"{path}/annotate_{ra_task.get_dataset_name()}_{task}_train_{split}.csv")
        else:
            ra_task.set_annotated_file_path(f"{path}/annotate_{ra_task.get_dataset_name()}_{task}_test_{split}.csv")
        try:
            ra_task._annotate(model=ra_model_finetuned, split=split)
            return jsonify({
                "dataset_path": ra_task.get_dataset_path(),
                "task": "Row Annotation",
                "status": "succes",
                "code": 200
            })
        except Exception as e:
            print(e)
            return jsonify({
                "message": "Error during the process:",
            }), 400
    elif task.lower() == 'cpa':
        cpa_task = CPATask(
            dataset_name=dataset_name,
            output_dataset=dataset_path
        )
        if not os.path.exists(path):
            path = os.makedirs(path)
        if is_train:
            cpa_task.set_annotated_file_path(f"{path}/annotate_{cpa_task.get_dataset_name()}_{task}_train_{split}.csv")
        else:
            cpa_task.set_annotated_file_path(f"{path}/annotate_{cpa_task.get_dataset_name()}_{task}_test_{split}.csv")
        cpa_task._annotate(model=cpa_model_finetuned, split=split, is_entity=is_entity)
        try:
            return jsonify({
                "dataset_path": cpa_task.get_dataset_path(),
                "task": "Column Property Annotation",
                "status": "succes",
                "code": 200
            })
        except Exception as e:
            print(e)
            return jsonify({
                "message": "Error during the process:",
            }), 400
    elif task.lower() == "cea":
        is_symbolic = request.json.get('is_symbolic', False)
        is_connectionist = request.json.get('is_connectionist', True)
        is_context = request.json.get('is_context', False)
        cea_task = CEATask(
            dataset_name=dataset_name,
            output_dataset=dataset_path
        )
        if not os.path.exists(path):
            path = os.makedirs(path)
        if is_train:
            cea_task.set_annotated_file_path(f"{path}/annotate_{cea_task.get_dataset_name()}_{task}_train_{split}.csv")
        else:
            cea_task.set_annotated_file_path(f"{path}/annotate_{cea_task.get_dataset_name()}_{task}_test_{split}.csv")
        cea_task._annotate(
            model=cea_model_finetuned_4,
            path=dataset_path,
            split=split,
            is_symbolic=is_symbolic,
            is_connectionist=is_connectionist,
            is_context=is_context
        )
        try:
            return jsonify({
                "dataset_path": cea_task.get_dataset_path(),
                "task": "Cell Entity Annotation",
                "status": "succes",
                "code": 200
            })
        except Exception as e:
            print(e)
            return jsonify({
                "message": "Error during the process:",
            }), 400
    elif task.lower() == "cta":
        is_combine_approach = request.json.get('is_combine_approach', False)
        cta_task = CTATask(
            dataset_name=dataset_name,
            output_dataset=dataset_path
        )
        if not os.path.exists(path):
            path = os.makedirs(path)
        if is_train:
            cta_task.set_annotated_file_path(f"{path}/annotate_{cta_task.get_dataset_name()}_{task}_train_{split}.csv")
        else:
            cta_task.set_annotated_file_path(f"{path}/annotate_{cta_task.get_dataset_name()}_{task}_test_{split}.csv")
        cta_task._annotate(
            model=cpa_model_finetuned,
            path=dataset_path,
            split=split,
            is_combine_approach=is_combine_approach
        )
        try:
            return jsonify({
                "dataset_path": cta_task.get_dataset_path(),
                "task": "Cell Entity Annotation",
                "status": "succes",
                "code": 200
            })
        except Exception as e:
            print(e)
            return jsonify({
                "message": "Error during the process:",
            }), 400
    elif task.lower() == "td":
        td_task = TDTask(
            dataset_name=dataset_name,
            output_dataset=dataset_path
        )
        if not os.path.exists(path):
            path = os.makedirs(path)
        if is_train:
            td_task.set_annotated_file_path(f"{path}/annotate_{td_task.get_dataset_name()}_{task}_train_{split}.csv")
        else:
            td_task.set_annotated_file_path(f"{path}/annotate_{td_task.get_dataset_name()}_{task}_test_{split}.csv")
        
        if is_entity:
            td_task._annotate(
                model=cpa_model_finetuned,
                path=dataset_path,
                split=split
            )
        else:
            td_task._annotate(
                model=cpa_model_finetuned,
                path=dataset_path,
                split=split
            )
        try:
            return jsonify({
                "dataset_path": td_task.get_dataset_path(),
                "task": "Cell Entity Annotation",
                "status": "succes",
                "code": 200
            })
        except Exception as e:
            print(e)
            return jsonify({
                "message": "Error during the process:",
            }), 400
    else:
        return jsonify({"message": "task does not exist"}), 400
    
# Route to annote tabular data
@app.route('/evaluate', methods=['POST'])
def evaluate():
    # Fill form to make dataset
    task = request.json.get('task', "")
    target_path = request.json.get('target_path', "")
    soumission_path = request.json.get('soumission_path', "")
    is_entity = request.json.get('is_entity', True)
    _client_payload = {}
    # End Form
    if task.lower() == "ra":
        return jsonify({
            "task": "Row Annotation",
            "status": "succes",
            "code": 200
        })
    elif task.lower() == 'cpa':
        
        _client_payload["submission_file_path"] = soumission_path
        aicrowd_evaluator = CPA_Evaluator(target_path, is_entity=is_entity)  # ground truth
        result = aicrowd_evaluator._evaluate(_client_payload)
        try:
            return jsonify({
                "metric": result,
                "task": "Column Property Annotation",
                "status": "succes",
                "code": 200
            })
        except Exception as e:
            print(e)
            return jsonify({
                "message": "Error during the process:",
            }), 400
    elif task.lower() == "cea":
        return jsonify({"message": task}), 200
    elif task.lower() == "cta":
        return jsonify({"message": task}), 200
    elif task.lower() == "td":
        return jsonify({"message": task}), 200
    else:
        return jsonify({"message": "task does not exist"}), 400
    

if __name__ == '__main__':
    app.run(debug=True, port=5001)
