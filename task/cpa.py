import json
import multiprocessing
import re
import numpy
import openai
from .helper import getNameCsvFile
import csv
import pandas as pd
import random,math
from .symbolic.api import get_properties, check_entity_properties_cpa
import concurrent.futures
import time

random.seed(42)
# from symbolic.api import openUrl
from .symbolic.api import openUrl

class CPATask:

    def  __init__(
        self, 
        output_dataset,
        raw_output_dataset, 
        target_file, table_path, 
        file_annotated,
        target_file_to_annotate
    ):
        """_summary_

        Args:
            output_dataset (file): csv dataset file final
            raw_output_dataset (file): raw csv dataset final that contain all cell in table without considering
            target.
            target_file (file): target csv file
            table_path (path): list of tables to make dataset
            file_annotated (file): the path of annotatedfile
        """
        self.output_dataset = output_dataset
        self.raw_output_dataset = raw_output_dataset
        self.target_file = target_file
        self.table_path = table_path
        self.file_annotated = file_annotated
        self.context_length = 4096
        self.target_file_to_annotate = target_file_to_annotate
        
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
                            result.append({"item": item, "index": idx, "identifier": identifier})
                        else:
                            result.append({"item": item, "index": idx, "data": temp})
            else:
                # Pour les itérations suivantes
                new_items = []
                for idx in nan_indices:
                    if not pd.isna(sublist[idx]):
                        if is_horizontal:
                            new_items.append({"item": sublist[idx], "index": idx, "identifier": identifier})
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
        list_file = getNameCsvFile(path=self.target_file)
        # open output_dataset cea file to write inside 
        with open(self.raw_output_dataset, "w+") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            # writer.writerow(header_cea)
            # get filename from each file in dataset
            for filed in list_file:
                filed += ".csv"
                if filed.endswith(".csv"):
                    filename = filed.split(".")[0]
                    if not header:
                        _file = pd.read_csv(f"{self.table_path}/{filed}", header=None)
                    else:
                        _file = pd.read_csv(f"{self.table_path}/{filed}", na_values=[''], keep_default_na=False)
                    
                    # get total row and colums of each cleaned file csv
                    
                    list_cell_selected = []
                    list_row__with_empty_data = []
                    if is_entity:
                        for index, row in _file.iloc[0:].iterrows():
                            print(row)
                            noNanElemenent = row[1:]
                            noNanElemenent = [str(x) for x in noNanElemenent if not isinstance(x, float) or not math.isnan(x)]
                            """ When file come from entity folder each cell represent column"""
                            _cell_selected = noNanElemenent
                            _cell_selected =  [random.choice(x.split(',')) for x in _cell_selected]
                            list_cell_selected.append(",".join(_cell_selected))
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
                        print(filename,total_col, len(list_cell_selected))
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
                            if total_col - len(list_cell_selected) > 1:
                                print(filename, total_col, len(list_cell_selected))
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
        return self.raw_output_dataset, self.target_file
    
    def _makeDataset(
        self,
        header=True,
        is_entity=True,
        is_horizontal=False
    ):
        """ 
            This function take two csv file which are almost same and compare the rows of the two files
            in order to create a new file that is same of the csv file 1
        """
        _raw_dataset, _target = self.buildDataset(
            header=header,
            is_entity=is_entity,
            is_horizontal=is_horizontal
        )
        # _raw_dataset, _target = self.raw_output_dataset, self.target_file
        with open(_target, 'r') as file1, open(_raw_dataset, 'r') as file2:
            _reader1 = csv.reader(file1)
            _reader2 = csv.reader(file2)
            csv.field_size_limit(200000)
            csv1_data = [row for row in _reader1]
            csv2_data = [row for row in _reader2] 
            
            updated_data = []
            if is_entity:
                updated_data.append(["tab_id", "col_j", "col_label", "list_col"])
            else:
                updated_data.append(["tab_id", "col0", "col_j", "col_label", "list_col"])
            for row1 in csv1_data:
                match_found = False
                for row2 in csv2_data:
                    if is_entity:
                        if row1[:2] == row2[:2]:
                            match_found = True
                            updated_data.append(row2)
                            # print(f"Row {row1} it is in CSV2")
                            break    
                    else:
                        if row1[:3] == row2[:3]:
                            match_found = True
                            updated_data.append(row2)
                            # print(f"Row {row1} it is in CSV2")
                            break      
                if match_found == False:
                    print(f"Row {row1} it is not in CSV2")
            
            with open(self.output_dataset, 'w', newline='') as updated_file:
                writer = csv.writer(updated_file)
                writer.writerows(updated_data)       
            print("Comparison completed. Updated CSV2 saved as 'updated_csv2.csv'.")
            
    def is_number(self, string):
        return re.match(r"^[+-]?\d+(\.\d+)?$", str(string)) is not None
    
    def is_date(self, string):
        split_slash = len(str(string.split("/")))
        split_tiret = len(str(string.split("-")))
        if split_slash == 3 or split_tiret == 3:
            if self.is_number(split_tiret[0]) or self.is_number(split_tiret[0][1:]) or  self.is_number(split_slash[0]):
                return True
        return False  
    
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
        try:
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
                
            # Update conversation
            conversation.append({"role": "assistant", "content": chat_response})
            try:
                
                result = json.loads(chat_response)
                label = result['recording']
                uri = result['uri']
                # print(f"The wikidata ofthis recording {label} is {uri}")
            except:
                uri = chat_response.split(":")[-1]
                uri = "http:" + uri.split('"')[0]
                print(chat_response)
                if len(uri.lower().split("q")) != 2:
                    return ""
                # print(f"The wikidata of this recording is {uri}")
        except:
            uri = "NIL"
        return uri
    def correct_spelling(self, text):
        prompt = f"Don't argument in your answer. Correct the spelling of this text :\"{text}\""
        model = "gpt-4o"
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
    def is_numerical(self, object: list):
        new_object = []
        for item in object:
            if isinstance(item, int) or isinstance(item, float):
                item = str(item)
                if "-" in item:
                    print(item)
                    new_object.append(item)
                else:
                    item = f"+{item}"
                    item = item.split(".")
                    if len(item) == 2:
                        if item[1] == "0":
                            item = item[0]
                        else:
                            item = ".".join(item)
                    else:
                        item = "".join(item)
                    new_object.append(item)
            else:
                new_object.append(item)
        return new_object
    
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
    
    def _annotate(self, model, split=0, is_entity=False, is_horizontal=False):
        filed = self.output_dataset

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
