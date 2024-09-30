import json
import multiprocessing
import re
import openai, os
from .helper import getNameCsvFile
import csv
import pandas as pd
import random
from .symbolic.api import get_properties, check_entity_properties_cpa
from .utils import *

random.seed(42)
# from symbolic.api import openUrl
from .symbolic.api import openUrl

class CPATask:

    def  __init__(
        self, 
        dataset_name = "cpa",
        output_dataset = None,
        target_file = None, 
        table_path = None, 
        file_annotated = None,
        target_file_gt = None
    ):
        """_summary_

        Args:
            dataset_name (_type_): _description_
            output_dataset (_type_, optional): _description_. Defaults to None.
            target_file (_type_, optional): _description_. Defaults to None.
            table_path (_type_, optional): _description_. Defaults to None.
            file_annotated (_type_, optional): _description_. Defaults to None.
            target_file_gt (_type_, optional): _description_. Defaults to None.
        """
        self.output_dataset = output_dataset
        self.target_file = target_file
        self.table_path = table_path
        self.file_annotated = file_annotated
        self.context_length = 4096
        self.target_file_gt = target_file_gt
        self.dataset_name = dataset_name
    def get_dataset_name(self):
        return self.dataset_name
        
    def get_dataset_path(self):
        return self.output_dataset
    
    def get_annotated_file(self):
        return self.file_annotated
    
    def set_annotated_file_path(self, path):
        self.file_annotated = path
        return self.file_annotated
    
    def set_dataset_path(self, dataset_path):
        self.output_dataset = dataset_path
        return self.output_dataset
    
    def set_target_file_path(self, path):
        """_summary_

        Args:
            path (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.target_file = path
        return self.target_file
    
    def set_table_path(self, path):
        """_summary_

        Args:
            path (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.table_path = path
        return self.table_path
      
    def openCSV(self, path):
        """ 
            path: path to csv file
        """
        df = pd.read_csv(path)
        # print(df['label'][0:10])       
        return df
    
    def process_data(self, data, is_horizontal=False):
        result = []
        nan_indices = []

        for i, sublist in enumerate(data):
            identifier = sublist[0]
            temp = [x for x in sublist if not pd.isna(x)]

            # Trouver les indices des valeurs nan dans la première itération
            if i == 0:
                nan_indices = [idx for idx, item in enumerate(sublist) if pd.isna(item)]
                for idx, item in enumerate(sublist):
                    if not pd.isna(item):
                        if is_horizontal:
                            result.append({"item": item, "index": idx, "data": identifier})
                        else:
                            result.append({"item": item, "index": idx, "data": temp})
            else:
                # Pour les itérations suivantes
                new_items = []
                for idx in nan_indices:
                    if not pd.isna(sublist[idx]):
                        if is_horizontal:
                            new_items.append({"item": sublist[idx], "index": idx, "data": identifier})
                        else:
                            new_items.append({"item": sublist[idx], "index": idx, 'data': temp})
                # Ajouter les nouveaux éléments non NaN à result
                if new_items:
                    result.extend(new_items)
                    # Mettre à jour les indices nan restants
                    nan_indices = [idx for idx in nan_indices if pd.isna(sublist[idx])]
                    if not nan_indices:
                        break

        # Trier le résultat par ordre de la clé 'index'
        result = sorted(result, key=lambda x: x['index'])

        return result
    
    
    def buildDataset(
        self,
        header=True,
        is_entity=True,
        is_horizontal=False
    ):
        """_summary_

        Args:
            header (bool, optional): true if table have header and False else. Defaults to True.
            col_before_row (bool, optional): true if the structure of target is 
            [tab_id col_id row_id]. Defaults to True.
            comma_in_cell (bool, optional): True if the data in a cell are consider as
            multi entities. Defaults to False.

        Returns:
            file: return target and raw dataset
        """
        # get name csv file inside of target_output_dataset without duplication
        if self.target_file_gt:
            list_file = getNameCsvFile(path=self.target_file_gt)
        elif self.target_file:
            list_file = getNameCsvFile(path=self.target_file)
        else:
            list_file = os.listdir(self.table_path)
        # open output_dataset cea file to write inside 
        with open(self.output_dataset, "w+") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            # writer.writerow(header_cea)
            # get filename from each file in dataset
            for filed in list_file:
                if filed.endswith(".csv"):
                    filename = filed.split(".")[0]
                    print(filename)
                    if not header:
                        _file = pd.read_csv(f"{self.table_path}/{filed}", header=None)
                    else:
                        _file = pd.read_csv(f"{self.table_path}/{filed}", na_values=[''], keep_default_na=False)
                    
                    # get total row and colums of each cleaned file csv
                    
                    list_cell_selected = []
                    list_row__with_empty_data = []
                    if is_entity:
                        for index, row in _file.iloc[0:].iterrows():
                            noNanElemenent = row[1:]
                            # print(noNanElemenent)
                            new_list = []
                            for elt in noNanElemenent:
                                # index = noNanElemenent.index(elt)
                                if pd.isna(elt) or contains_html_tags(str(elt)):
                                    continue
                                elif "http" and "%" in str(elt):
                                    new_element = decole_url_file(str(elt))
                                    # noNanElemenent[index] = new_element
                                    new_list.append(new_element)
                                else:
                                    new_list.append(elt)
                            noNanElemenent = new_list
                            # noNanElemenent = [str(x) for x in noNanElemenent if (not isinstance(x, float) ) and not contains_html_tags(str(x))]
                            """ When file come from entity folder each cell represent column"""
                            _cell_selected = noNanElemenent
                            _cell_selected =  [random.choice(str(x).split(',')) for x in _cell_selected]
                            list_cell_selected.append(",".join(_cell_selected))
                            # print(list_cell_selected)
                    else:
                        
                        # print("fichier:", filename, "Number of cols ", total_col)
                        noNanElemenent = []
                        indices = []
                        if not is_horizontal:
                            total_col = len(_file.axes[1])
                        else:
                            total_col = len(_file.axes[1])
                        for index, row in _file.iterrows():
                            if not is_horizontal:
                                for x in row:
                                    if not pd.isna(x):
                                        noNanElemenent.append(x)
                                if len(noNanElemenent) == total_col:
                                    list_cell_selected = noNanElemenent
                                    list_row__with_empty_data = []
                                    break
                                else:   
                                    list_row__with_empty_data.append(row.values.tolist())
                                    continue
     
                            else:
                                for x in row[1:]:
                                    if not pd.isna(x):
                                        noNanElemenent.append(random.choice(str(x).split(",")))
                                if len(noNanElemenent) == total_col -1:
                                    list_cell_selected = noNanElemenent
                                    list_row__with_empty_data = []
                                    break
                                else:
                                    temp = []
                                    for item in row[1:]:
                                        if not pd.isna(item):
                                            choice = random.choice(str(item).split(","))
                                            while choice == 'nan':
                                                print('===========YES+++++++++====')
                                                choice = random.choice(str(item).split(","))                 
                                            temp.append(choice)
                                        else:
                                            temp.append(item)
                                    list_row__with_empty_data.append(temp)
                                    noNanElemenent = []  
                                    continue      
                    if is_entity:
                        col = 0
                        print(len(list_cell_selected))
                        while col < len(list_cell_selected):
                            writer.writerow([filename, col+1, list_cell_selected[col], list_cell_selected])
                            col += 1
                            
                    else:
                        # print(filename,total_col, len(list_cell_selected))
                        if list_cell_selected:
                            col = 0
                            if not is_horizontal:
                                total_col = len(list_cell_selected) -1
                                while col < total_col:
                                    writer.writerow([filename, 0, col+1, list_cell_selected[col+1], list_cell_selected])
                                    col += 1
                            else:
                                total_col = len(list_cell_selected)
                                while col < total_col:
                                    writer.writerow([filename, 0, col+1, list_cell_selected[col], list_cell_selected])
                                    col += 1
                        else:
                            list_cell_selected = self.process_data(list_row__with_empty_data, is_horizontal=is_horizontal)
                            # if total_col - len(list_cell_selected) > 1:
                            #     print(filename, total_col, len(list_cell_selected))
                            col = 0
                            if not is_horizontal:
                                total_col = len(list_cell_selected) -1
                                while col < total_col:
                                    writer.writerow([filename, 0, col+1, list_cell_selected[col+1]['item'], list_cell_selected[col+1]['data']])
                                    col += 1
                            else:                
                                total_col = len(list_cell_selected)
                                while col < total_col:
                                    writer.writerow([filename, 0, col+1, list_cell_selected[col]['item'], list_cell_selected[col]['data']])
                                    col += 1

                                
                else:
                    print("it is not csv file")
            csv_file.close()
        return self.output_dataset
    
    def _makeDataset(
        self,
        header=True,
        is_entity=True,
        is_horizontal=False,
        is_train=True,
    ):
        """ 
            This function take two csv file which are almost same and compare the rows of the two files
            in order to create a new file that is same of the csv file 1
        """
        _raw_dataset = self.buildDataset(
            header=header,
            is_entity=is_entity,
            is_horizontal=is_horizontal
        )
        # _raw_dataset, = self.output_dataset
        """
            function returns a new dataset if the target file exists else returns datasets already built
        """
        if self.target_file_gt:
            with open(self.target_file_gt, 'r') as file1, open(_raw_dataset, 'r') as file2:
                _reader1 = csv.reader(file1)
                _reader2 = csv.reader(file2)
                csv.field_size_limit(200000)
                csv1_data = [row for row in _reader1]
                csv2_data = [row for row in _reader2] 
                
                updated_data = []
                if is_entity:
                    updated_data.append(["tab_id", "col_j", "col_label", "list_col", "entity"])
                else:
                    updated_data.append(["tab_id", "col0", "col_j", "col_label", "list_col", "entity"])
                for row1 in csv1_data:
                    match_found = False
                    for row2 in csv2_data:
                        if is_entity:
                            if row1[:2] == row2[:2]:
                                match_found = True
                                if is_train:
                                    row2.append(row1[2])
                                else:
                                    row2.append("NIL")
                                updated_data.append(row2)
                                # print(f"Row {row1} it is in CSV2")
                                break    
                        else:
                            if row1[:3] == row2[:3]:
                                match_found = True
                                if is_train:
                                    row2.append(row1[3])
                                else:
                                    row2.append("NIL")
                                updated_data.append(row2)
                                # print(f"Row {row1} it is in CSV2")
                                break      
                    if match_found == False:
                        print(f"Row {row1} it is not in CSV2")
                
                with open(self.output_dataset, 'w', newline='') as updated_file:
                    writer = csv.writer(updated_file)
                    writer.writerows(updated_data)       
                print("Comparison completed. Updated CSV2 saved as 'updated_csv2.csv'.")
        else:
            df = pd.read_csv(self.output_dataset)
            datas = df.values.tolist()
            with open(self.output_dataset, 'w', newline='') as updated_file:
                writer = csv.writer(updated_file)
                writer.writerows(datas)
            return self.output_dataset
    
    def _csv_to_jsonl(self, csv_path, json_path):
        """ 
            csv_path: path to csv file
            json_path: path to json file
        """
        df = self.openCSV(csv_path)
        datas = []
        for i in range(len(df['col_label'])):
            # print(df['tab_id'][i])
            value_property = ""
            try:         
                value_property = df['col_label'][i]
                value_property = correct_string(str(value_property))
                value_property = value_property.split("::")[0].strip('"').strip(" ")
            except:
                value_property = [df['col_label'][i]]
                value_property = correct_string(value_property)
                value_property = value_property.split("::")[0].strip('"').strip(" ")
            uri = df['entity'][i]
            
            prompt_input = f"Please which wikidata property has this value: {value_property}"
            if len(prompt_input) >= 2048:
                print(len(prompt_input))
            datas.append(
                {
                    "messages":  [
                        {
                            "role": "system", 
                            "content": "Hi, I'm semantic annotation Agent. What can i do to help you today."
                        },
                        {
                            "role": "user", 
                            "content": prompt_input
                        },      
                        {
                            "role": "assistant", 
                            "content": f"""{{"value_property":  "{value_property}", "uri": "{uri}"}}"""
                        }
                    ]
                }
            )
 
        print(datas[:3])
        with open(json_path, 'w') as f:
            for data in datas:
                json.dump(data, f)
                f.write('\n')
        
        return datas
            
    
    def compute_max_token(self, prompt_length, max_new_token):
        max_returned_tokens = max_new_token + prompt_length
        # print("Prompt length:", prompt_length)
        assert max_returned_tokens <= self.context_length, (
            max_returned_tokens,
            self.context_length
        )
    
    def inference(self, model_id, user_input=None, temperature=0.82, frequency_penalty=0, presence_penalty=0, max_tokens=256):
        chatBot = "Hi, I'm semantic annotation Agent. What can i do to help you today."
        if user_input is None:
            user_input = input('User: \n')
        conversation = [{"role": "user", "content": user_input}]
        
        prompt_length = len(user_input)
        # check the total length contex
        self.compute_max_token(prompt_length=prompt_length, max_new_token=max_tokens)
        # print(user_input)
        message_input = conversation.copy()
        prompt = [{"role": "system", "content": chatBot}]
        message_input.insert(0, prompt[0])
        completion = openai.chat.completions.create(
            model=model_id,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            top_p=1,
            presence_penalty=presence_penalty,
            seed=42,
            messages=message_input,
            max_tokens=max_tokens
        )
        # Extract the chatbot response from API response
        chat_response = completion.choices[0].message.content
        # chat_response = correct_expression(chat_response)
        # Update conversation
        conversation.append({"role": "assistant", "content": chat_response})
        print(conversation)
        try:
            result = json.loads(chat_response)
            uri = result['uri']
            return uri
        except:
            result = chat_response.split('uri": ')
            uri = result[-1].strip('"').split('}')[0].strip(" ").strip('"')
        return uri
    
    def correct_spelling(self, text):
        prompt = f"Don't argument in your answer. Correct the spelling of this text :\"{text}\""
        model = "gpt-4o-mini"
        message_input = [{"role": "user", "content": prompt}]
        completion = openai.chat.completions.create(
            model=model,
            temperature=0.8,
            top_p=1,
            messages=message_input,
            max_tokens=256,
            seed=42
        )
        corrected_text = completion.choices[0].message.content.split('"')
        if len(corrected_text) == 1:
            corrected_text = "".join(corrected_text).strip()
        elif len(corrected_text) <=3:
            corrected_text = corrected_text[1]
            corrected_text = f"{corrected_text}.".strip(".").strip()
        else:
            corrected_text = corrected_text[3].split(".")[0].strip()
        print(f"The correct spelling of {text} is {corrected_text}")
        return corrected_text
    
    def get_property_entities(self, entity_id=None, prop_values=None, label=None, is_horizontal=False, index=None, target_label=None):
        # if not self.is_number(label) or not self.is_date(label):     
        #     label = self.correct_spelling(label)
        # if not self.is_number(target_label) or not self.is_date(target_label):       
        #     target_label = self.correct_spelling(label)
        if is_horizontal:
            # label it is already the ids of the recording of the row
            result = check_entity_properties_cpa([entity_id], prop_values, is_horizontal=is_horizontal, label_value=label, index=index)
        else:
            entity_ids = openUrl(label, prop_values, is_horizontal)
            result = check_entity_properties_cpa(entity_ids, prop_values, is_horizontal=is_horizontal, label_value=target_label)

        return result
    
    def run_with_timeout(self, entity_id, context, label, is_horizontal, index, timeout=8):
        def target(result_queue, *args):
            result = self.get_property_entities(*args)
            result_queue.put(result)
        
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=target, args=(result_queue, entity_id, context, label, is_horizontal, index))
        
        process.start()
        process.join(timeout)
        
        if process.is_alive():
            process.terminate()
            process.join()
            return None
        else:
            return result_queue.get()
    
    def display_result(self, result, target_data, i, updated_cpa_data):
        for j in range(0, len(result)):
            prop_id = result[j]['prop_id']
            prop_label = result[j]['prop_value']
            print("*************************")
            print(f"col {j+i} annotated")
            print(f"label col={prop_label}, => prop_id={prop_id}")
            print("*************************\n")
            data = target_data[j+i]
            data.append(prop_id)
            updated_cpa_data.append(data)
        return updated_cpa_data
    
    # specifici case concerning 
    
    def _annotate_symbolique(self, model, path=None, split=0, is_entity=False, is_horizontal=False):
        if not path:
            filed = self.output_dataset
        else:
            filed = path

        with open(self.target_file_to_annotate, "r") as csv_file:
            target_reader = csv.reader(csv_file)
            target_data = [row for row in target_reader]
            with open(self.file_annotated, 'w', newline='') as updated_file:
                writer = csv.writer(updated_file)
                # writer.writerow(header_cea)
                # check if it is csv file
                if filed.endswith(".csv"):
                    print(filed)
                    i = split
                    # print(i, len(target_data[split:]))
                    _file = pd.read_csv(filed, na_values=[''], keep_default_na=False) # open file with pandas
                    while i < len(target_data):
                        updated_cea_data = []   # at each iteration in reader_data, empty the list
                        print(i)
                        if not is_entity and not is_horizontal:
                            context = eval(_file['list_col'][i])
                            target_label = _file['col_label'][i]
                            result = ''
                            if context:
                                print(context)
                                # format properties
                                new_context = self.is_numerical(context)
                                label = new_context[0]
                                if self.is_number(label):
                                    if "+" in label:
                                        label = label[1:]
                                context = new_context
                                print(context)
                                print("target_label",target_label)
                                if len(context) == 1:
                                    result = self.get_property_entities(label=label, prop_values=context, target_label=target_label)
                                else:
                                    result = self.get_property_entities(label=label, prop_values=context[1:], target_label=target_label)             
                          
                            if result:
                                updated_cea_data = self.display_result([result], target_data, i, updated_cea_data)
                                writer.writerows(updated_cea_data)
                                i += 1
                            else:
                                if not self.is_number(label) and not self.is_date(label):
                                    old_label = label # store the old label
                                    label = self.correct_spelling(label)
                                    if len(context) == 1:
                                        result = self.get_property_entities(label=label, prop_values=context, target_label=target_label)
                                    else:
                                        result = self.get_property_entities(label=label, prop_values=context[1:], target_label=target_label)
                                    # check the result with new label
                                    print(result)
                                    if result:
                                        updated_cea_data = self.display_result([result], target_data, i, updated_cea_data)
                                        writer.writerows(updated_cea_data)
                                        i += 1
                                    else:
                                        # try it by spelling the target_col_label
                                        if not self.is_number(target_label) and not self.is_date(target_label):
                                            target_label = self.correct_spelling(target_label)
                                            if len(context) == 1:
                                                result = self.get_property_entities(label=label, prop_values=context, target_label=target_label)
                                            else:
                                                result = self.get_property_entities(label=label, prop_values=context[1:], target_label=target_label)
                                            # check if result
                                            if result:
                                                updated_cea_data = self.display_result([result], target_data, i, updated_cea_data)
                                                writer.writerows(updated_cea_data)
                                                i += 1
                                            else:
                                                if len(context) == 1:
                                                    result = self.get_property_entities(label=old_label, prop_values=context, target_label=target_label)
                                                else:
                                                    result = self.get_property_entities(label=old_label, prop_values=context[1:], target_label=target_label)
                                                if result:
                                                    updated_cea_data = self.display_result([result], target_data, i, updated_cea_data)
                                                    writer.writerows(updated_cea_data)
                                                    i += 1
                                                else:                                           
                                                    data = target_data[i]
                                                    data.append("NIL")
                                                    print("*************************")
                                                    print(f"col {i} annotated")
                                                    print(f"indentifier={label}, target_col={target_label} => prop_id=NIL")
                                                    print("*************************\n")
                                                    updated_cea_data.append(data)
                                                    i += 1
                                                    writer.writerows(updated_cea_data)
                                                    
                                        else:
                                            data = target_data[i]
                                            data.append("NIL")
                                            print("*************************")
                                            print(f"col {i} annotated")
                                            print(f"indentifier={label}, target_col={target_label} => prop_id=NIL")
                                            print("*************************\n")
                                            updated_cea_data.append(data)
                                            i += 1
                                            writer.writerows(updated_cea_data)
                                else:
                                    # at this level the identifier is a number or date
                                    #  That means that the target_col is misspelled:
                                    if not self.is_number(target_label) and not self.is_date(target_label):
                                        target_label = self.correct_spelling(target_label)
                                        if len(context) == 1:
                                            result = self.get_property_entities(label=label, prop_values=context, target_label=target_label)
                                        else:
                                            result = self.get_property_entities(label=label, prop_values=context[1:], target_label=target_label)
                                        if result:
                                            updated_cea_data = self.display_result([result], target_data, i, updated_cea_data)
                                            writer.writerows(updated_cea_data)
                                            i += 1
                                        else:
                                            data = target_data[i]
                                            data.append("NIL")
                                            print("*************************")
                                            print(f"col {i} annotated")
                                            print(f"indentifier={label}, target_col={target_label} => prop_id=NIL")
                                            print("*************************\n")
                                            updated_cea_data.append(data)
                                            i += 1
                                            writer.writerows(updated_cea_data)
                                    else:
                                        data = target_data[i]
                                        data.append("NIL")
                                        print("*************************")
                                        print(f"col {i} annotated")
                                        print(f"indentifier={label}, target_col={target_label} => prop_id=NIL")
                                        print("*************************\n")
                                        updated_cea_data.append(data)
                                        i += 1
                                        writer.writerows(updated_cea_data)

                        elif is_horizontal:
                            context = eval(_file['list_col'][i])
                            label = _file['col_label'][i]
                            index = context.index(label)
                            if isinstance(label, str):
                                label = label.strip()
                            # if self.is_number(label):
                            #     if label.split(".")[-1] == "0":
                            #         label = label.split(".")[0]
                            #         label = f"+{label}"
                            #         print(label)
                                
                            user_input = f"Please Give the  wikidata URI of the this recording: {context}"
                            uri = self.inference(model, user_input=user_input)
                            entity_id = uri.split('/')[-1]
                            print(label, entity_id)
                            result = self.run_with_timeout(entity_id, context, label, is_horizontal, index)
                            if result:
                                updated_cea_data = self.display_result([result], target_data, i, updated_cea_data)
                                writer.writerows(updated_cea_data)
                                i += len([result])
                            else:
                                result = get_properties(context, is_horizontal, label, index)
                                if result:
                                    updated_cea_data = self.display_result([result], target_data, i, updated_cea_data)
                                    writer.writerows(updated_cea_data)
                                    # i += len(result)
                                    i += 1
                                else:
                                    data = target_data[i]
                                    data.append("NIL")
                                    updated_cea_data.append(data)
                                    writer.writerows(updated_cea_data)
                                    print("*************************")
                                    print(f"col {i} annotated")
                                    print(f"label col={label}, => prop_id=NIL")
                                    print("*************************\n")
                                    i += 1
                        else:
                            
                            values = eval(_file['list_col'][i])
                            result = get_properties(values=values)
                            if result:
                                updated_cea_data = self.display_result(result, target_data, i, updated_cea_data)
                                writer.writerows(updated_cea_data)
                                i += len(result)
                            else:
                                for val in values:
                                    data = target_data[i]
                                    data.append("NIL")
                                    updated_cea_data.append(data)
                                    i += 1
                                writer.writerows(updated_cea_data)
                        
                                                                      
                    else:
                        print("it is not csv file")

    def _annotate(self, model, path=None, split=0, is_entity=False):
        """_summary_

        Args:
            model (_type_): _description_
            path (_type_, optional): _description_. Defaults to None.
            split (int, optional): _description_. Defaults to 0.
        """
        if not path:
            dataset_path = self.output_dataset
        else:
            dataset_path = path
        if self.target_file_gt:
            with open(self.target_file_gt, "r") as csv_file:
                target_reader = csv.reader(csv_file)
                target_data = [row for row in target_reader]
                with open(self.file_annotated, 'w', newline='') as updated_file:
                    writer = csv.writer(updated_file)
                    # writer.writerow(header_cea)
                    # check if it is csv file
                    if dataset_path.endswith(".csv"):
                        print(dataset_path)
                        df = pd.read_csv(dataset_path) # open file with pandas
                        i = split
                        for data in target_data[split:]:
                            updated_cea_data = []   # at each iteration in reader_data, empty the list
                            label =  df['col_label'][i]    
                            user_input = f"Please which wikidata property has this valuee: {label}"                
                            # check uri
                            result = self.inference(model_id=model, user_input=user_input)                                 
                            # add result of annation   
                            data.append(result)
                            updated_cea_data.append(data)
                                
                            #  write data in update cea file
                            writer.writerows(updated_cea_data)
                            result = result.split("/")[-1]
                            print("*************************")
                            print(f"col {i} annotated")
                            print(f"label col={label}, => prop_id={result}")
                            print("*************************\n")
                            i += 1  
                                        
                        else:
                            print("it is not csv file")
        else:
            with open(self.file_annotated, 'w', newline='') as updated_file:
                writer = csv.writer(updated_file)
                # writer.writerow(header_cea)
                # check if it is csv file
                if dataset_path.endswith(".csv"):
                    print(dataset_path)
                    df = pd.read_csv(dataset_path) # open file with pandas
                    datas = df.values.tolist()
                    i = split
                    for data in datas[split:]:
                        # print(data)
                        updated_data = []   # at each iteration in reader_data, empty the list
                        label =  df['col_label'][i]
                        # label = correct_json_string(label).strip(" ")
                        # label = label.split("::")[0].strip('"').strip(" ")
        
                        user_input = f"Please which wikidata property has this valuee: {label}"                
                        # check uri
                        result = self.inference(model_id=model, user_input=user_input)                                 
                        # add result of annation   
                        data.append(result)
                        if is_entity:
                            # tab_id, col_i, property
                            updated_data.extend([data[0], data[1], data[-1]])
                        else:
                            # tab_id, col0, colj, property
                            updated_data.extend([data[0], data[1], data[2], data[-1]])
                        #  write data in update cea file
                        print(updated_data)
                        writer.writerow(updated_data)
                        print("*************************")
                        print(f"col {i} annotated")
                        print(f"label col={label}, => prop_id={result.split("/")[-1]}")
                        print("*************************\n")               
                        i += 1                              
                    else:
                        print("it is not csv file")