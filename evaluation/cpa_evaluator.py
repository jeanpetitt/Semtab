import pandas as pd
class CPA_Evaluator:
  def __init__(self, answer_file_path, round=1, is_entity=False):
    """
    `round` : Holds the round for which the evaluation is being done. 
    can be 1, 2...upto the number of rounds the challenge has.
    Different rounds will mostly have different ground truth files.
    """
    self.answer_file_path = answer_file_path
    self.round = round
    self.is_entity = is_entity

  def _evaluate(self, client_payload):
    """
    `client_payload` will be a dict with (atleast) the following keys :
      - submission_file_path : local file path of the submitted file
      - aicrowd_submission_id : A unique id representing the submission
      - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
    """
    submission_file_path = client_payload["submission_file_path"]

    gt_cols_pro = dict()
    if self.is_entity:
        gt = pd.read_csv(self.answer_file_path, delimiter=',', names=['tab_id', 'obj_col_id', 'property'],
                        dtype={'tab_id': str, 'obj_col_id': str, 'property': str}, keep_default_na=False)
        for index, row in gt.iterrows():
            cols = '%s %s' % (row['tab_id'], row['obj_col_id'])
            gt_cols_pro[cols] = row['property']

        annotated_cols, correct_cols = set(), set()
        sub = pd.read_csv(submission_file_path, delimiter=',', names=['tab_id', 'obj_col_id', 'property'],
                        dtype={'tab_id': str, 'obj_row_id': str, 'property': str}, keep_default_na=False)
        for index, row in sub.iterrows():
            cols = '%s %s' % (row['tab_id'], row['obj_col_id'])
            if cols in gt_cols_pro:
                if cols in annotated_cols:
                    # continue
                    raise Exception("Duplicate column pairs in the submission file")
                else:
                    annotated_cols.add(cols)
                annotation = row['property']
                if not annotation.startswith('http://www.wikidata.org/prop/direct/'):
                    annotation = 'http://www.wikidata.org/prop/direct/' + annotation
                if annotation.lower() == gt_cols_pro[cols].lower():
                    correct_cols.add(cols)
    else:
        gt = pd.read_csv(self.answer_file_path, delimiter=',', names=['tab_id', 'sub_col_id', 'obj_col_id', 'property'],
                        dtype={'tab_id': str, 'sub_col_id': str, 'obj_col_id': str, 'property': str}, keep_default_na=False)
        for index, row in gt.iterrows():
            cols = '%s %s %s' % (row['tab_id'], row['sub_col_id'], row['obj_col_id'])
            gt_cols_pro[cols] = row['property']

        annotated_cols, correct_cols = set(), set()
        sub = pd.read_csv(submission_file_path, delimiter=',', names=['tab_id', 'sub_col_id', 'obj_col_id', 'property'],
                        dtype={'tab_id': str, 'sub_col_id': str, 'obj_row_id': str, 'property': str}, keep_default_na=False)
        for index, row in sub.iterrows():
            cols = '%s %s %s' % (row['tab_id'], row['sub_col_id'], row['obj_col_id'])
            if cols in gt_cols_pro:
                if cols in annotated_cols:
                    # continue
                    raise Exception("Duplicate column pairs in the submission file")
                else:
                    annotated_cols.add(cols)
                annotation = row['property']
                if not annotation.startswith('http://www.wikidata.org/prop/direct/'):
                    annotation = 'http://www.wikidata.org/prop/direct/' + annotation
                if annotation.lower() == gt_cols_pro[cols].lower():
                    correct_cols.add(cols)
    print(f"correct cpa: {len(correct_cols)}, Annotated cpa: {len(annotated_cols)}, target cpa: {len(gt_cols_pro)}")
    precision = float(len(correct_cols)) / len(annotated_cols) if len(annotated_cols) > 0 else 0.0
    recall = float(len(correct_cols)) / len(gt_cols_pro.keys())
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    main_score = f1
    secondary_score = precision
    print('%.3f %.3f %.3f' % (f1, precision, recall))

    """
    Do something with your submitted file to come up
    with a score and a secondary score.

    if you want to report back an error to the user,
    then you can simply do :
      `raise Exception("YOUR-CUSTOM-ERROR")`

     You are encouraged to add as many validations as possible
     to provide meaningful feedback to your users
    """
    _result_object = {
        "score": main_score,
        "score_secondary": secondary_score
    }
    return _result_object