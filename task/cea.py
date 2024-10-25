from .symbolic.api import openUrl, check_entity_properties_cea
from .helper import getNameCsvFile, getAllCellInTableColByBCol, getAllCellInTableRowByRow, tableToVectROw, tableToVectCol
import random
import csv
import json
import re
import os
import pandas as pd
import numpy as np
import openai
from .utils import *
# from symbolic.api import openUrl

class CEATask:

    def  __init__(
        self, 
        dataset_name = "cea",
        output_dataset = None,
        target_file = None, 
        table_path = None, 
        file_annotated = None,
        target_file_gt = None
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
    
    def buildDataset(
        self,
        header,
        col_before_row,
        comma_in_cell,
        transpose,
        is_entity
    ):
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
            for filed in list_file:
                if filed.endswith(".csv"):
                    filename = filed.split(".")[0]
                    if not header:
                        _file = pd.read_csv(f"{self.table_path}/{filed}", header=None)
                    else:
                        _file = pd.read_csv(f"{self.table_path}/{filed}")
                    
                    # get total row and colums of each cleaned file csv
                    total_rows = len(_file.axes[0])
                    total_cols=len(_file.axes[1])
                    print("fichier:", filename, "nbre de ligne: ", total_rows, " nombre de col: ", total_cols)
                    
                    if transpose:
                        # get all the cell in a table
                        list_cell = getAllCellInTableRowByRow(
                            table=_file, 
                            comma_in_cell=comma_in_cell,
                        )
                        
                        tableToVectROw(
                            table=_file, 
                            writer=writer, 
                            list_cell=list_cell, 
                            filename=filename, 
                            col_before_row=col_before_row
                        )
                    else:
                        # get all the cell in a table
                        list_cell = getAllCellInTableColByBCol(
                            table=_file, 
                            is_entity=is_entity, 
                            comma_in_cell=comma_in_cell
                        )
                        
                        tableToVectCol(
                            table=_file, 
                            writer=writer, 
                            list_cell=list_cell, 
                            filename=filename, 
                            col_before_row=col_before_row
                        )
                    
                else:
                    print("it is not csv file")
                    csv_file.close()
        return self.output_dataset

    
    def _makeDataset(
        self,
        header=True,
        col_before_row=True,
        comma_in_cell=False,
        transpose=False,
        is_train=True,
        is_entity=False,
        split=0
    ):
        """_summary_

        Args:
            header (bool, optional): _description_. Defaults to True. Change it to False if your tables don't have header
            col_before_row (bool, optional): _description_. Defaults to True. If this args is set to true, 
                it allow to determine if the output will be (tab_id, col_id, row_id) rather than (tab_id, row_id, col_id)
            comma_in_cell (bool, optional): _description_. Defaults to False. Specify if the tables dataset has at least 2 entities in the table split by the comma(,)
            transpose (bool, optional): _description_. Defaults to False. If this argument is set to True, table2Vect will transform table in vector row by row rather than col by col else.
            is_vertical (bool, optional): _description_. Defaults to False. Set it to True if the type of tables is horizontal.
            is_train (bool, optional): _description_. Defaults to False. Set it to True if it is a train or validation dataset.
            split (int, optional): _description_. Defaults to 0. detemine at which position you want ta start compare the dataset with target file
        """
        _raw_dataset = self.buildDataset(
            header=header,
            col_before_row=col_before_row,
            comma_in_cell=comma_in_cell,
            transpose=transpose,
            is_entity=is_entity
        )
        # _raw_dataset, _target = self.raw_output_dataset, self.target_file
        csv.field_size_limit(20000000)
        if self.target_file_gt:
            with open(self.target_file_gt, 'r') as file1, open(_raw_dataset, 'r') as file2:
                _reader1 = csv.reader(file1)
                _reader2 = csv.reader(file2)
                csv1_data = [row for row in _reader1]
                csv2_data = [row for row in _reader2]     
                with open(self.output_dataset, 'w', newline='') as updated_file:
                    writer = csv.writer(updated_file)
                    
                    
                    updated_data = []

                    if col_before_row == True:
                        updated_data.extend(["tab_id", "col_id", "row_id","label", 'context', 'entity'])
                    else:
                        updated_data.extend(["tab_id", "row_id", "col_id","label", 'context', 'entity'])
                    writer.writerows([updated_data])
                    for row1 in csv1_data[split:]:
                        match_found = False
                        for row2 in csv2_data:
                            if row1[:3] == row2[:3]:
                                match_found = True
                                if is_train:
                                    row2.append(row1[3])
                                else:
                                    row2.append("NIL")
                                updated_data.append(row2)
                                writer.writerow(row2)
                                break         
                        if not match_found:
                            print(f"Row {row1} it is not in CSV2")
        else:
            df = pd.read_csv(self.output_dataset)
            return df
            
        print(f"Comparison completed. Updated CSV2 saved {self.output_dataset}")

    def _csv_to_jsonl(self, csv_path, json_path, is_entity=False, comma_in_cell=False):
        """ 
            csv_path: path to csv file
            json_path: path to json file
        """
        df = self.openCSV(csv_path)
        datas = []
        
        for i in range(len(df['label'])):
            if df['label'][i].split(","):     
                uri_list = str(df['entity'][i]).split(",")
                if not comma_in_cell:
                    label = df['label'][i]
                    uri = df['entity'][i]
                    context = eval(df['context'][i]) if ("[" and ']') in df['context'][i] else df['context'][i]
                    for item in context:
                        index = context.index(item)
                        if ',' in item:
                            item = item.split(",")
                            if label in item:
                                context[index] = label
                    datas.append(
                        {
                            "messages":  [
                                {
                                    "role": "system", 
                                    "content": "Semantic Table Interpretation"
                                },
                                {
                                    "role": "user", 
                                    "content": f"Please What is wikidata URI of {label}.\nContext: {context}",
                                },      
                                {
                                    "role": "assistant", 
                                    "content": f"{{\"entity\": {label}, \"context\": {context}, \"uri\": \"{uri}\"}}"
                                }
                            ]
                        }
                    )
                else:
                    j = 0
                    label_list = df['label'][i].split(",")
                    if len(uri_list) != len(label_list):
                        raise ValueError("error", df['tab_id'][i])
                    for uri in uri_list:
                        label = correct_string(str(label_list[j]))
                        context = eval(df['context'][i]) if ("[" and ']') in df['context'][i] else df['context'][i]
                        if is_entity:
                            datas.append(
                                {
                                    "messages":  [
                                        {
                                            "role": "system", 
                                            "content": "Semantic Table Interpretation"
                                        },
                                        {
                                            "role": "user", 
                                            "content": f"Please What is wikidata URI of {label}.\nContext: {context}",
                                        },      
                                        {
                                            "role": "assistant", 
                                            "content": f"{{\"entity\": {label}, \"context\": {context}, \"uri\": \"{uri}\"}}"
                                        }
                                    ]
                                }
                            )
                        else:       
                            datas.append(
                                {
                                    "messages":  [
                                        {
                                            "role": "system", 
                                            "content": "Semantic Table Interpretation"
                                        },
                                        {
                                            "role": "user", 
                                            "content": f"Please What is wikidata URI of {label_list[j]}.\nContext: {context}",
                                        },      
                                        {
                                            "role": "assistant", 
                                            "content": f"\"entity\": \"{label_list[j]}\", \"context\": \"{context}\", \"uri\": \"{uri}\""
                                        }
                                    ]
                                }
                            )
                        j += 1
                
            else:
                print("not split available")
        
        print(datas[0])
        with open(json_path, 'w') as f:
            for data in datas:
                json.dump(data, f)
                f.write('\n')
        
        return datas
    
    def _combineJsonFile(self, json_path: str, json_output_dataset: str, split: str):
        """_summary_

        Args:
            json_path (folder): path of the folder that contain json file
            json_output_dataset (file): output_dataset json file obtained after combine file
            split (str): example train[:10%], test[:10%], val[:10%]
        
        """
        
        files = os.listdir(json_path)
        datas = []
        data_size = 0
        number_sample = 0
        # get the percentage of the dataset 
        num_split = re.findall(r'\d+', split)
        percent = int(num_split[0])
        
        with open(json_output_dataset, 'w') as f:
            for _file in files:
                with open(f"{json_path}/{_file}", 'r') as json_file:
                    for line in json_file:
                        # f.write(line)
                        datas.append(line)
            data_size = len(datas)
            if split.startswith("train"):     
                number_sample = float((data_size * percent) / 100)
                # get the integer and decimal part of the number
                number_sample = str(number_sample).split(".")
                integer, decimal = number_sample[0], f"0.{number_sample[-1]}"
                
                # get round up or round down
                if float(decimal) >= 0.5:
                    number_sample = int(integer) + 1
                else:
                    number_sample = int(integer)
                
                train = datas[:number_sample]
                for data in train:
                    f.write(data)
                return train
            
            elif split.startswith("val"):     
                number_sample = float((data_size * percent) / 100)
                # get the integer and decimal part of the number
                number_sample = str(number_sample).split(".")
                integer, decimal = number_sample[0], f"0.{number_sample[-1]}"
                
                # get round up or round down
                if float(decimal) >= 0.5:
                    number_sample = int(integer) + 1
                else:
                    number_sample = int(integer)
                
                validation = datas[:number_sample]
                for data in validation:
                    f.write(data)
                
                
                return validation
            
            elif split.startswith("test"):     
                number_sample = float((data_size * percent) / 100)
                # get the integer and decimal part of the number
                number_sample = str(number_sample).split(".")
                integer, decimal = number_sample[0], f"0.{number_sample[-1]}"
                
                # get round up or round down
                if float(decimal) >= 0.5:
                    number_sample = int(integer) + 1
                else:
                    number_sample = int(integer)
                
                test = datas[:number_sample]
                for data in test:
                    f.write(data)
                return test
    
    def compute_max_token(self, prompt_length, max_new_token):
        max_returned_tokens = max_new_token + prompt_length
        print("Prompt length:", prompt_length)
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
        print(user_input)
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
                
                # result = json.loads(chat_response)
                result = eval(chat_response)
                label = result['label']
                uri = result['uri']
                print(f"The wikidata of {label} is {uri}")
            except:
                uri = chat_response.split(":")[-1]
                uri = "http:" + uri.split('"')[0]
                print(chat_response)
                print(f"The wikidata of this entity is {uri}")
        except openai.InternalServerError as e:
            print(e)
            uri = "NIL"
        return uri
    def is_number(self, string):
        return re.match(r"^[+-]?\d+(\.\d+)?$", str(string)) is not None
    
    def is_date(self, string):
        split_slash = len(string.split("/"))
        split_tiret = len(string.split("-"))
        if split_slash == 3 or split_tiret == 3:
            if self.is_number(split_tiret[0]) or self.is_number(split_tiret[0][1:]) or  self.is_number(split_slash[0]):
                return False
        return True
    
    def choose_random_valid_element(self,my_list):
        valid_elements = [element for element in my_list if element not in (None, "", False)]
        if not valid_elements:
            return None

        return random.choice(valid_elements)
    
    def correct_spelling(self, text):
        prompt = f"Don't argument in your answer. Correct the spelling of this text : \"{text}\""
        model = "gpt-4o-mini"
        message_input = [{"role": "user", "content": prompt}]
        completion = openai.chat.completions.create(
            model=model,
            temperature=0.8,
            top_p=1,
            seed=42,
            messages=message_input,
            max_tokens=256
        )
        
        corrected_text = completion.choices[0].message.content.split('"')
        print(corrected_text)
        if len(corrected_text) == 1:
            corrected_text = "".join(corrected_text).strip()
        elif len(corrected_text) <=3:
            corrected_text = corrected_text[1]
            corrected_text = f"{corrected_text}.".strip(".").strip()
        else:
            corrected_text = corrected_text[3].split(".")[0].strip()
        return corrected_text
    
    
    def _annotate(
        self,
        model,
        path=None,
        split=0,
        is_symbolic=False,
        is_connectionist=True,
        is_context=False
        ):
        """_summary_

        Args:
            model (_type_): _description_
            path (_type_, optional): _description_. Defaults to None.
            comma_in_cell (bool, optional): _description_. Defaults to False.
            col_before_row (bool, optional): _description_. Defaults to True.
            split (int, optional): _description_. Defaults to 0.
            is_symbolic (bool, optional): _description_. Defaults to False.
            is_context (bool, optional): _description_. Defaults to False.
            is_connectionist (bool, optional): _description_. Defaults to True.
        """
        if not path:
            dataset_path = self.output_dataset
        else:
            dataset_path = path
        
        # if col_before_row:
        #     header_cea = ["tab_id", "col_id", "row_id", "entity"]
        # else:
        #     header_cea = ["tab_id", "row_id", "col_id", "entity"]
            
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
                        df = pd.read_csv(dataset_path, dtype=str) # open file with pandas
                        i = split
                        for data in target_data[split:]:
                            updated_cea_data = []   # at each iteration in reader_data, empty the list
                            label =  df['label'][i]     
            
                            if type(label) == type(np.nan):
                                result = "NIL"
                            else:
                                # get annotation of the cell
                                uri = []
                                result = "NIL"
                                if is_connectionist and is_symbolic:
                                    user_input = f"Please what is wikidata URI of {label} entity"
                                    uri = result = self.inference(model_id=model, user_input=user_input)
                                    if uri:
                                        result = uri
                                    else:
                                        uri = openUrl(str(label).strip(".").strip())
                                        if uri:
                                            result = uri
                                        else:                                         
                                            label = self.correct_spelling(label)
                                            uri = openUrl(label)
                                            if uri:
                                                result = uri
                                            else:
                                                result = "NIL"
                                else:
                                    # use symbolic approach
                                    if is_symbolic:
                                        new_context = []
                                        if df['context'][i].lower() != 'nil':
                                            context = eval(df['context'][i])
                                            print(label)
                                            if isinstance(context, list):
                                                for item in context:
                                                    if isinstance(item, int) or isinstance(item, float):
                                                        if "-" in str(item):
                                                            new_context.append(str(item))
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
                                                            new_context.append(item)
                                                    else:
                                                        new_context.append(item)
                                            else:
                                                new_context = [context]
                                        else:
                                            context = ''
                                            is_context = False
                                        print(context)
                                        context = new_context
                                        context_have_string_value = False
                                        for lab in context[1:]:
                                                if isinstance(lab, str):
                                                    if lab[0:4].isdigit() or "".join(lab[1:].split('.')).isdigit():
                                                        continue
                                                    context_have_string_value = True
                                                    break                   
                                        print(context)
                                        old_label = label
                                        if is_context:
                                            entity_ids = openUrl(label, context[1:])
                                            if not entity_ids:
                                                label = self.correct_spelling(label)
                                                entity_ids = openUrl(label)
                                                if not entity_ids:
                                                    result = ''
                                                elif len(entity_ids) == 1:
                                                    result = entity_ids[0]
                                                else:
                                                    if label not in context and label != old_label and old_label not in context:
                                                        context = [label]
                                                    elif label not in context and label != old_label:
                                                        index = context.index(old_label)
                                                        context[index] = label
                                                    if context.index(label) != 0:
                                                        entity_ids = openUrl(context[0], context[1:])
                                                        result = check_entity_properties_cea(entity_ids, context[1:], False, label, context_have_string_value)
                                                    else:
                                                        result = check_entity_properties_cea(entity_ids, context[1:], context_have_string_value=context_have_string_value)
                                            else:
                                                if len(entity_ids) == 1:
                                                    result = entity_ids[0]
                                                else:
                                                    if label not in context:
                                                        first_elt = context[0]
                                                        context.pop()
                                                        context.extend([label, first_elt])
                                                    if context.index(label) != 0:
                                                        entity_ids = openUrl(context[0], context[1:])
                                                        result = check_entity_properties_cea(entity_ids, context[1:], False, label, context_have_string_value)
                                                    else:
                                                        result = check_entity_properties_cea(entity_ids, context[1:], context_have_string_value=context_have_string_value)
                                            print(f"Label = {label}, context={context}, -> target_uri = http://www.wikidata.org/entity/{result}")
                                        else:
                                            result = openUrl(label)
                                    
                                    else:  
                                        # Use connectionist approach               
                                        user_input = f"Please what is wikidata URI of {label} entity.\nContext: {df['context'][i]}"            
                                        if len(user_input) > 200:
                                            user_input = f"Please what is wikidata URI of {label}"
                                            # check uri
                                        result = self.inference(model_id=model, user_input=user_input)
                                        if result == "http:":
                                            print("result nil. try again")
                                            user_input = f"Please what is wikidata URI of {label} entity" 
                                            result = self.inference(model_id=model, user_input=user_input)
   
                            data.append(result)
                            updated_cea_data.append(data)
                            i += 1                             
                            #  write data in update cea file
                            writer.writerow(updated_cea_data)
                            print("*************************")
                            print(f"Cell {i} annotated")
                            print("*************************")
                        
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
                    print(len(df))
                    i = 0
                    for data in datas[split:]:
                        updated_cea_data = []   # at each iteration in reader_data, empty the list
                        label =  df['label'][i]       
                        if not label:
                            result = "NIL"
                        else:
                            # get annotation of the cell
                            uri = []
                            result = "NIL"
                            if is_connectionist and is_symbolic:
                                user_input = f"Please what is wikidata URI of {label} entity.\nContext: {df['context'][i]}"            
                                if len(user_input) > 200:
                                    user_input = f"Please what is wikidata URI of {label}"
                                # check uri
                                uri = self.inference(model_id=model, user_input=user_input)
                                if result == "http:":
                                    print("result nil. try again")
                                    user_input = f"Please what is wikidata URI of {label} entity" 
                                    uri = self.inference(model_id=model, user_input=user_input)
                                if uri:
                                    result = uri
                                else:
                                    uri = openUrl(str(label).strip(".").strip())
                                    if uri:
                                        result = uri
                                    else:                                         
                                        label = self.correct_spelling(label)
                                        uri = openUrl(label)
                                        if uri:
                                            result = uri
                                        else:
                                            result = "NIL"
                            else:
                                # use symbolic approach
                                if is_symbolic:
                                    new_context = []
                                    if df['context'][i].lower() != 'nil':
                                        context = eval(df['context'][i])
                                        print(label)
                                        if isinstance(context, list):
                                            for item in context:
                                                if isinstance(item, int) or isinstance(item, float):
                                                    if "-" in str(item):
                                                        new_context.append(str(item))
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
                                                        new_context.append(item)
                                                else:
                                                    new_context.append(item)
                                        else:
                                            new_context = [context]
                                    else:
                                        context = ''
                                        is_context = False
                                    print(context)
                                    context = new_context
                                    context_have_string_value = False
                                    for lab in context[1:]:
                                        if isinstance(lab, str):
                                            if lab[0:4].isdigit() or "".join(lab[1:].split('.')).isdigit():
                                                continue
                                            context_have_string_value = True
                                            break                   
                                    print(context)
                                    old_label = label
                                    if is_context:
                                        entity_ids = openUrl(label, context[1:])
                                        if not entity_ids:
                                            label = self.correct_spelling(label)
                                            entity_ids = openUrl(label)
                                            if not entity_ids:
                                                result = ''
                                            elif len(entity_ids) == 1:
                                                result = entity_ids[0]
                                            else:
                                                if label not in context and label != old_label and old_label not in context:
                                                    context = [label]
                                                elif label not in context and label != old_label:
                                                    index = context.index(old_label)
                                                    context[index] = label
                                                if context.index(label) != 0:
                                                    entity_ids = openUrl(context[0], context[1:])
                                                    result = check_entity_properties_cea(entity_ids, context[1:], False, label, context_have_string_value)
                                                else:
                                                    result = check_entity_properties_cea(entity_ids, context[1:], context_have_string_value=context_have_string_value)
                                        else:
                                            if len(entity_ids) == 1:
                                                result = entity_ids[0]
                                            else:
                                                if label not in context:
                                                    first_elt = context[0]
                                                    context.pop()
                                                    context.extend([label, first_elt])
                                                if context.index(label) != 0:
                                                    entity_ids = openUrl(context[0], context[1:])
                                                    result = check_entity_properties_cea(entity_ids, context[1:], False, label, context_have_string_value)
                                                else:
                                                    result = check_entity_properties_cea(entity_ids, context[1:], context_have_string_value=context_have_string_value)
                                        print(f"Label = {label}, context={context}, -> target_uri = http://www.wikidata.org/entity/{result}")
                                    else:
                                        result = openUrl(label)                       
                                else:  
                                    # Use connectionist approach               
                                    user_input = f"Please what is wikidata URI of {label} entity.\nContext: {df['context'][i]}"            
                                    # if len(user_input) > 200:
                                    #     user_input = f"Please what is wikidata URI of {label}"
                                        # check uri
                                    result = self.inference(model_id=model, user_input=user_input)
                                    if result == "http:":
                                        print("result nil. try again")
                                        user_input = f"Please what is wikidata URI of {label} entity" 
                                        result = self.inference(model_id=model, user_input=user_input)
                            
                        data.append(result)
                        updated_cea_data.extend([data[0], data[1], data[2], data[-1]])
                        i += 1  
                                
                        #  write data in update cea file
                        writer.writerows([updated_cea_data])
                        print("*************************")
                        print(f"Cell {i} annotated")
                        print("*************************")
                        
                else:
                    print("it is not csv file")
