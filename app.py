from flask import Flask, request, jsonify
from task.ra import RATask
from task.cpa import CPATask
import os
from path_data.utils import ra_model_finetuned, cpa_model_finetuned

app = Flask(__name__)

# Simulons un dataset pour l'exemple
dataset = []
# Route pour générer un dataset
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
    # End Form
    if task.lower() == "ra":
        ra_task = RATask(
            dataset_name=dataset_name,
            table_path=table_path,
            target_file=target_path, 
            target_file_gt=target_gt_path
        )
        if not os.path.exists("results/dataset"):
            os.makedirs("results/dataset")
        if is_train:
            ra_task.set_dataset_path(f"results/dataset/dataset_{ra_task.get_dataset_name()}_{task}_train.csv")
        else:
            ra_task.set_dataset_path(f"results/dataset/dataset_{ra_task.get_dataset_name()}_{task}_test.csv")
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
        if not os.path.exists("results/dataset"):
            os.makedirs("results/dataset")
        if is_train:
            cpa_task.set_dataset_path(f"results/dataset/dataset_{cpa_task.get_dataset_name()}_{task}_train.csv")
        else:
            cpa_task.set_dataset_path(f"results/dataset/dataset_{cpa_task.get_dataset_name()}_{task}_test.csv")
        try:
            cpa_task._makeDataset(header=header, is_train=is_train, is_horizontal=is_horizontal, is_entity=is_entity)
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
        return jsonify({"message": task}), 200
    elif task.lower() == "cta":
        return jsonify({"message": task}), 200
    elif task.lower() == "td":
        return jsonify({"message": task}), 200
    else:
        return jsonify({"message": "task does not exist"}), 400
    

# Route pour annoter les données
@app.route('/annotate', methods=['POST'])
def annotate():
    # Fill form to make dataset
    task = request.json.get('task', "")
    dataset_name = request.json.get('dataset_name', "tsotsa")
    dataset_path = request.json.get('dataset_path', "")
    is_train = request.json.get('table_path', True)
    split = request.json.get('split', 0)
    is_entity = request.json.get('is_entity', False)
    # End Form
    if task.lower() == "ra":
        ra_task = RATask(
            dataset_name=dataset_name,
            output_dataset=dataset_path
        )
        if not os.path.exists("results/annotate"):
            os.makedirs("results/annotate")
        if is_train:
            ra_task.set_annotated_file_path(f"results/annotate/annotate_{ra_task.get_dataset_name()}_{task}_train.csv")
        else:
            ra_task.set_annotated_file_path(f"results/annotate/annotate_{ra_task.get_dataset_name()}_{task}_test.csv")
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
        if not os.path.exists("results/annotate"):
            os.makedirs("results/annotate")
        if is_train:
            cpa_task.set_annotated_file_path(f"results/annotate/annotate_{cpa_task.get_dataset_name()}_{task}_train.csv")
        else:
            cpa_task.set_annotated_file_path(f"results/annotate/annotate_{cpa_task.get_dataset_name()}_{task}_test.csv")
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
        return jsonify({"message": task}), 200
    elif task.lower() == "cta":
        return jsonify({"message": task}), 200
    elif task.lower() == "td":
        return jsonify({"message": task}), 200
    else:
        return jsonify({"message": "task does not exist"}), 400
    

if __name__ == '__main__':
    app.run(debug=True, port=5000)
