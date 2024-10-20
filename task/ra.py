from .helper import getNameCsvFile
import csv, json, os, re
import pandas as pd
import numpy as np
import openai
import random,math
from .utils import *

random.seed(42)
# from symbolic.api import openUrl
from .symbolic.api import openUrl

class RATask:

    def  __init__(
        self, 
        dataset_name = "ra",
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
    
    def buildDataset(
        self,
        header=True,
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
                    if not header:
                        _file = pd.read_csv(f"{self.table_path}/{filed}", header=None)
                    else:
                        _file = pd.read_csv(f"{self.table_path}/{filed}")
                    
                    # get total row and colums of each cleaned file csv
                    total_row = len(_file.axes[0])
                    print("fichier:", filename, "Number of rows ", total_row)
                    list_cell_selected = []
                    for index, row in _file.iloc[0:].iterrows():
                        noNanElemenent = row[1:]
                        noNanElemenent = [str(x) for x in noNanElemenent if (not isinstance(x, float) and not is_date(str(x)) and not is_number(str(x)) and is_valid_string(str(x)) and not contains_html_tags(str(x)) and not ('http' and "%") in str(x))]
                        # print(noNanElemenent)
                        # noNanElemenent = ','.join(noNanElemenent)
                        # noNanElemenent = noNanElemenent.split(",")
                        
                        """ 
                            Each group of cells selected represent the elements that describe the best 
                            the column
                        """
                        if len(noNanElemenent) > 15:
                            _cell_selected = random.sample(noNanElemenent, k=10)
                            _cell_selected = [random.choice(x.split(',')) for x in _cell_selected]
                            # print(_cell_selected)
                            list_cell_selected.append(_cell_selected)
                        else:
                            _cell_selected = noNanElemenent
                            _cell_selected =  [random.choice(x.split(',')) for x in _cell_selected]
                            # print(_cell_selected)
                            list_cell_selected.append(_cell_selected)
                    row = 0
                    while row < total_row:
                        writer.writerow([filename, row, list_cell_selected[row]])
                        row += 1
                else:
                    print("it is not csv file")
            csv_file.close()
        return self.output_dataset
    
    def _makeDataset(
        self,
        header=True,
        is_train=True,
        split=0
    ):
        """ _summary_
                This function take two csv file which are almost same and compare the rows of the two files
                in order to create a new file that is same of the csv file 1
        Args:
            header(str): Optional
        """
        _raw_dataset = self.buildDataset(
            header=header
        )
        # _raw_dataset = self.raw_output_dataset
        if self.target_file_gt:
            with open(self.target_file_gt, 'r') as file1, open(_raw_dataset, 'r') as file2:
                _reader1 = csv.reader(file1)
                _reader2 = csv.reader(file2)
                
                csv1_data = [row for row in _reader1]
                csv2_data = [row for row in _reader2]     
                
            with open(self.output_dataset, 'w', newline='') as updated_file:
                writer = csv.writer(updated_file) 
                writer.writerow(["tab_id", "row_id", "record", "uri"])       
                for row1 in csv1_data[split:]:
                    # print(row1, self.target_file_gt, self.target_file)
                    match_found = False
                    for row2 in csv2_data:
                        if row1[:2] == row2[:2]:
                            match_found = True
                            if is_train:
                                row2.append(row1[2])
                            else:
                                row2.append("NIL")
                            writer.writerow(row2)  
                            # print(f"Row {row1} it is in CSV2")
                            break         
                    if match_found == False:
                        print(f"Row {row1} it is not in CSV2")      
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
            Args:
                csv_path: path to csv file
                json_path: path to json file
        """
        df = self.openCSV(csv_path)
        datas = []
        
        for i in range(len(df['record'])):
            list_entity = eval(df['record'][i])
            list_entity = ','.join(list_entity)
            list_entity = list_entity.split(",")
            uri = df['uri'][i]
          
            datas.append(
                {
                    "messages":  [
                        {
                            "role": "system", 
                            "content": "Hi, I'm semantic annotation Agent. What can i do to help you today."
                        },
                        {
                            "role": "user", 
                            "content": f"Please Give the  wikidata URI of the this recording: {list_entity}"
                        },      
                        {
                            "role": "assistant", 
                            "content": f"""{{"recording": {list_entity}, "uri": "{uri}"}}"""
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
        """_summary_

        Args:
            prompt_length (int): _description_
            max_new_token (int): _description_
        """
        max_returned_tokens = max_new_token + prompt_length
        print("Prompt length:", prompt_length)
        assert max_returned_tokens <= self.context_length, (
            max_returned_tokens,
            self.context_length
        )
    
    def inference(self, model_id, user_input=None, temperature=0.82, frequency_penalty=0, presence_penalty=0, max_tokens=256):
        """_summary_

        Args:
            model_id (str): _description_
            user_input (str, optional): _description_. Defaults to None.
            temperature (float, optional): _description_. Defaults to 0.82.
            frequency_penalty (int, optional): _description_. Defaults to 0.
            presence_penalty (int, optional): _description_. Defaults to 0.
            max_tokens (int, optional): _description_. Defaults to 256.

        Returns:
            wikidata_uri: This function ruturn a wikidata uri for a given recording passed in input
        """
        chatBot = "Hi, I'm semantic annotation Agent. What can i do to help you today."
        if user_input is None:
            user_input = input('User: \n')
        conversation = [{"role": "user", "content": user_input}]
        
        try:
            prompt_length = len(user_input)
            # check the total length contex
            self.compute_max_token(prompt_length=prompt_length, max_new_token=max_tokens)
            print(user_input)
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
                
            # Update conversation
            conversation.append({"role": "assistant", "content": chat_response})
            try:
                
                result = json.loads(chat_response)
                label = result['recording']
                uri = result['uri']
                uri = uri.split(' ')[0]
                print(f"The wikidata ofthis recording {label} is {uri}")
            except:
                uri = chat_response.split(":")[-1]
                uri = "http:" + uri.split('"')[0]
                uri = uri.split(' ')[0]
                print(chat_response)
                print(f"The wikidata of this recording is {uri}")
        except:
            uri = "NIL"
        return uri
    
    
    def _annotate(self, model, path=None, split=0):
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
                            label =  df['record'][i]     
            
                            user_input = f"Please Give the  wikidata URI of the this recording: {label}"                
                            # check uri
                            result = self.inference(model_id=model, user_input=user_input)                                 
                            # add result of annation   
                            data.append(result)
                            updated_cea_data.append(data)
                            i += 1  
                                
                            #  write data in update cea file
                            writer.writerows(updated_cea_data)
                            print("*************************")
                            print(f"Row {i} annotated")
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
                    i = split
                    for data in datas[split:]:
                        # print(data)
                        updated_data = []   # at each iteration in reader_data, empty the list
                        label =  df['record'][i]     
        
                        user_input = f"Please Give the  wikidata URI of the this recording: {label}"                
                        # check uri
                        result = self.inference(model_id=model, user_input=user_input)                                 
                        # add result of annation   
                        data.append(result)
                        updated_data.extend([data[0], data[1], data[-1]])
                        i += 1                              
                        #  write data in update cea file
                        print(updated_data)
                        writer.writerow(updated_data)
                        print("*************************")
                        print(f"Row {i} annotated")
                        print("*************************")                
                    else:
                        print("it is not csv file")
