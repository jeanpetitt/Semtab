import pandas as pd


class TD_Evaluator:
    def __init__(self, answer_file_path, round=1):
        """
        `round` : Holds the round for which the evaluation is being done. 
        can be 1, 2...upto the number of rounds the challenge has.
        Different rounds will mostly have different ground truth files.
        """
        self.answer_file_path = answer_file_path
        self.round = round

    def _evaluate(self, client_payload):
        """
        `client_payload` will be a dict with (atleast) the following keys :
          - submission_file_path : local file path of the submitted file
        """
        submission_file_path = client_payload["submission_file_path"]

        gt_cell_ent = dict()
        gt = pd.read_csv(self.answer_file_path, delimiter=',', names=['tab_id', 'entity'],
                         dtype={'tab_id': str, 'entity': str}, keep_default_na=False)
        for index, row in gt.iterrows():
            cell = '%s' % (row['tab_id'])
            gt_cell_ent[cell] = row['entity'].lower().split(' ')

        correct_cells, annotated_cells = set(), set()
        sub = pd.read_csv(submission_file_path, delimiter=',', names=['tab_id', 'entity'],
                          dtype={'tab_id': str, 'entity': str}, keep_default_na=False)

        for index, row in sub.iterrows():
            cell = '%s' % (row['tab_id'])
            # print(cell)
            if cell in gt_cell_ent:
                if cell in annotated_cells:
                    raise Exception("Duplicate cells in the submission file")
                else:
                    annotated_cells.add(cell)

                annotation = row['entity'].lower()
                if annotation in gt_cell_ent[cell]:
                    correct_cells.add(cell)
                    # print(correct_cells)
        print("Correct row:", len(correct_cells), "row annotated:", len(annotated_cells), "target rows:", len(gt_cell_ent))
        precision = float(len(correct_cells)) / \
            len(annotated_cells) if len(annotated_cells) > 0 else 0.0
        recall = float(len(correct_cells)) / len(gt_cell_ent.keys())
        f1 = (2 * precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0.0
        main_score = f1
        secondary_score = precision

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
            "F1 score": main_score,
            "precision": secondary_score,
            "recall": recall
        }
        return _result_object
