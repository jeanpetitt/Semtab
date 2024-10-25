from flask import Flask, request, jsonify
from evaluation.cea_evaluator import CEA_Evaluator
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
from path_data.utils import ra_model_finetuned_1, ra_model_finetuned, cpa_model_finetuned, cta_model_finetuned_2, cea_model_finetuned_5, cea_model_finetuned_6, cea_model_finetuned_4, td_model_finetuned_2, td_model_finetuned_1

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
    split = request.json.get('split', 0)
    path = f"results/dataset"
    # End Form
    if task.lower() == "ra":
        ra_task = RATask(
            dataset_name=dataset_name,
            table_path=table_path,
            target_file=target_path, 
            target_file_gt=target_gt_path
        )
        if not os.path.exists(f"{path}/{task}"):
            os.makedirs(f"{path}/{task}")
            path = f"{path}/{task}"
        if is_train:
            ra_task.set_dataset_path(f"{path}/dataset_{ra_task.get_dataset_name()}_{task}_train_split_{split}.csv")
        else:
            ra_task.set_dataset_path(f"{path}/dataset_{ra_task.get_dataset_name()}_{task}_test_split_{split}.csv")
        try:
            ra_task._makeDataset(header=header, is_train=is_train, split=split)
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
        if not os.path.exists(f"{path}/{task}"):
            os.makedirs(f"{path}/{task}")
            path = f"{path}/{task}"
        if is_train:
            cpa_task.set_dataset_path(f"{path}/dataset_{cpa_task.get_dataset_name()}_{task}_train_split_{split}.csv")
        else:
            cpa_task.set_dataset_path(f"{path}/dataset_{cpa_task.get_dataset_name()}_{task}_test_split_{split}.csv")
        cpa_task._makeDataset(
            header=header,
            is_train=is_train,
            is_horizontal=is_horizontal,
            is_entity=is_entity,
            split=split
        )
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
        is_entity = request.json.get('is_entity', False)
        transpose = request.json.get('transpose', False)
        col_before_row = request.json.get('col_before_row', True)
        comma_in_cell = request.json.get('comma_in_cell', False)
        cea_task = CEATask(
            dataset_name=dataset_name,
            table_path=table_path,
            target_file=target_path, 
            target_file_gt=target_gt_path
        )
        if not os.path.exists(f"{path}/{task}"):
            os.makedirs(f"{path}/{task}")
            path = f"{path}/{task}"
        if is_train:
            cea_task.set_dataset_path(f"{path}/dataset_{cea_task.get_dataset_name()}_{task}_train_split_{split}.csv")
        else:
            cea_task.set_dataset_path(f"{path}/dataset_{cea_task.get_dataset_name()}_{task}_test_split_{split}.csv")
        cea_task._makeDataset(
            header=header, 
            is_entity=is_entity, 
            transpose=transpose, 
            col_before_row=col_before_row,
            comma_in_cell=comma_in_cell,
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
        cta_task = CTATask(
            dataset_name=dataset_name,
            table_path=table_path,
            target_file=target_path, 
            target_file_gt=target_gt_path
        )
        if not os.path.exists(f"{path}/{task}"):
            os.makedirs(f"{path}/{task}")
            path = f"{path}/{task}"
        if is_train:
            cta_task.set_dataset_path(f"{path}/dataset_{ra_task.get_dataset_name()}_{task}_train_split_{split}.csv")
        else:
            cta_task.set_dataset_path(f"{path}/dataset_{ra_task.get_dataset_name()}_{task}_test_split_{split}.csv")
        try:
            cta_task._makeDataset(header=header, is_train=is_train, split=split)
            return jsonify({
                "dataset_path": cta_task.get_dataset_path(),
                "table_path": cta_task.target_file_gt,
                "task": "Row Annotation",
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
            table_path=table_path,
            target_file=target_path, 
            target_file_gt=target_gt_path
        )
        if not os.path.exists(f"{path}/{task}"):
            os.makedirs(f"{path}/{task}")
            path = f"{path}/{task}"
        if is_train:
            td_task.set_dataset_path(f"{path}/dataset_{ra_task.get_dataset_name()}_{task}_train_split_{split}.csv")
        else:
            td_task.set_dataset_path(f"{path}/dataset_{ra_task.get_dataset_name()}_{task}_test_split_{split}.csv")
        try:
            td_task._makeDataset(header=header, is_train=is_train, split=split)
            return jsonify({
                "dataset_path": td_task.get_dataset_path(),
                "table_path": td_task.target_file_gt,
                "task": "Row Annotation",
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
@app.route('/annotate', methods=['POST'])
def annotate():
    # Fill form to make dataset
    task = request.json.get('task', "")
    dataset_name = request.json.get('dataset_name', "tsotsa")
    dataset_path = request.json.get('dataset_path', "")
    is_train = request.json.get('table_path', True)
    split = request.json.get('split', 0)
    is_entity = request.json.get('is_entity', False)
    path = f"results/annotate"
    # End Form
    if task.lower() == "ra":
        ra_task = RATask(
            dataset_name=dataset_name,
            output_dataset=dataset_path
        )
        if not os.path.exists(f"{path}/{task}"):
            os.makedirs(f"{path}/{task}")
        path = f"{path}/{task}"
        if is_train:
            ra_task.set_annotated_file_path(f"{path}/annotate_{ra_task.get_dataset_name()}_{task}_train_{split}.csv")
        else:
            ra_task.set_annotated_file_path(f"{path}/annotate_{ra_task.get_dataset_name()}_{task}_test_{split}.csv")
        ra_task._annotate(
            model=ra_model_finetuned_1,
            split=split,
            # path=dataset_path
        )
        try:
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
        if not os.path.exists(f"{path}/{task}"):
            os.makedirs(f"{path}/{task}")
        path = f"{path}/{task}"
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
        split_end = request.json.get('split_end', None)
        is_context = request.json.get('is_context', False)
        cea_task = CEATask(
            dataset_name=dataset_name,
            output_dataset=dataset_path
        )
        if not os.path.exists(f"{path}/{task}"):
            os.makedirs(f"{path}/{task}")
        path = f"{path}/{task}"
        if is_train:
            cea_task.set_annotated_file_path(f"{path}/annotate_{cea_task.get_dataset_name()}_{task}_train_{split}.csv")
        else:
            cea_task.set_annotated_file_path(f"{path}/annotate_{cea_task.get_dataset_name()}_{task}_test_{split}.csv")
        cea_task._annotate(
            model=cea_model_finetuned_6,
            path=dataset_path,
            split_start=split,
            split_end=split_end,
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
        if not os.path.exists(f"{path}/{task}"):
            os.makedirs(f"{path}/{task}")
        path = f"{path}/{task}"
        if is_train:
            cta_task.set_annotated_file_path(f"{path}/annotate_{cta_task.get_dataset_name()}_{task}_train_{split}.csv")
        else:
            cta_task.set_annotated_file_path(f"{path}/annotate_{cta_task.get_dataset_name()}_{task}_test_{split}.csv")
        cta_task._annotate(
            model=cta_model_finetuned_2,
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
        if not os.path.exists(f"{path}/{task}"):
            os.makedirs(f"{path}/{task}")
        path = f"{path}/{task}"
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
    submition_path = request.json.get('submition_path', "")
    is_entity = request.json.get('is_entity', True)
    _client_payload = {}
    # End Form
    if task.lower() == "ra":
        _client_payload["submission_file_path"] = submition_path
        aicrowd_evaluator = RA_Evaluator(target_path)  # ground truth
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
    elif task.lower() == 'cpa':
        
        _client_payload["submission_file_path"] = submition_path
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
        _client_payload["submission_file_path"] = submition_path
        aicrowd_evaluator = CEA_Evaluator(target_path)  # ground truth
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
    elif task.lower() == "cta":
        _client_payload["submission_file_path"] = submition_path
        aicrowd_evaluator = CTA_Evaluator(target_path)  # ground truth
        result = aicrowd_evaluator._evaluate_without_ancestor(_client_payload)
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
    elif task.lower() == "td":
        _client_payload["submission_file_path"] = submition_path
        aicrowd_evaluator = TD_Evaluator(target_path)  # ground truth
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
    else:
        return jsonify({"message": "task does not exist"}), 400
    
@app.route('/parse_csv_to_json', methods=['POST'])
def parse_csv_to_json():
     # Fill form to make dataset
    task = request.json.get('task', "cea")
    dataset_name = request.json.get('dataset_name', "")
    csv_path = request.json.get('csv_path', "")
    is_entity = request.json.get('is_entity', "")
    comma_in_cell = request.json.get('comma_in_cell', "")
    path = f"results/jsonl"
    # End Form
    if task.lower() == "ra":
        ra_task = RATask(
            dataset_name=dataset_name
        )

        if not os.path.exists(f"{path}/{task}"):
            os.makedirs(f"{path}/{task}")
            path = f"{path}/{task}"
        
        try:
            ra_task._csv_to_jsonl(
                csv_path=csv_path,
                json_path=f"{path}/{dataset_name}_{task}.jsonl"
            )
            return jsonify({
                "task": "Cell Entity Annotation",
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
            dataset_name=dataset_name
        )

        if not os.path.exists(f"{path}/{task}"):
            os.makedirs(f"{path}/{task}")
            path = f"{path}/{task}"
        
        try:
            cpa_task._csv_to_jsonl(
                csv_path=csv_path,
                json_path=f"{path}/{dataset_name}_{task}.jsonl"
            )
            return jsonify({
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
        cea_task = CEATask(
            dataset_name=dataset_name
        )

        if not os.path.exists(f"{path}/{task}"):
            os.makedirs(f"{path}/{task}")
            path = f"{path}/{task}"
        
        cea_task._csv_to_jsonl(
                csv_path=csv_path,
                json_path=f"{path}/{dataset_name}_{task}.jsonl",
                comma_in_cell=comma_in_cell,
                is_entity=is_entity
            )
        try:
            return jsonify({
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
        cta_task = CTATask(
            dataset_name=dataset_name
        )

        if not os.path.exists(f"{path}/{task}"):
            os.makedirs(f"{path}/{task}")
            path = f"{path}/{task}"
        
        try:
            cta_task._csv_to_jsonl(
                csv_path=csv_path,
                json_path=f"{path}/{dataset_name}_{task}.jsonl"
            )
            return jsonify({
                "task": "Column Type Annotation",
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
            dataset_name=dataset_name
        )

        if not os.path.exists(f"{path}/{task}"):
            os.makedirs(f"{path}/{task}")
            path = f"{path}/{task}"
        
        try:
            td_task._csv_to_jsonl(
                csv_path=csv_path,
                json_path=f"{path}/{dataset_name}_{task}.jsonl"
            )
            return jsonify({
                "task": "Table Detection",
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

if __name__ == '__main__':
    app.run(debug=True, port=5001)
