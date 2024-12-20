import random
import pandas as pd
import os, json, csv
from .helper import getNameCsvFile
import re, openai
from collections import Counter
from .utils import *

random.seed(42)
def get_most_common_element(lst):
    counter = Counter(lst)
    most_common = counter.most_common()
    if not lst:
        return None
    # check if element have the same frequence
    if len(set(counter.values())) == 1:
        return random.choice(lst)
    
    # get element that be more often repeat
    max_count = most_common[0][1]
    most_common_elements = [elem for elem, count in most_common if count == max_count]

    return random.choice(most_common_elements)

class TDTask:
    
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
    
    def compute_max_token(self, prompt_length, max_new_token):
        max_returned_tokens = max_new_token + prompt_length
        # print("Prompt length:", prompt_length)
        try:
            assert max_returned_tokens <= self.context_length, (
                max_returned_tokens,
                self.context_length
            )
            return max_returned_tokens
        except AssertionError as e:
            print(e)
    
    # def is_date(self, string):
    #     split_slash = len(str(string.split("/")))
    #     split_tiret = len(str(string.split("-")))
    #     if split_slash == 3 or split_tiret == 3:
    #         if self.is_number(split_tiret[0]) or self.is_number(split_tiret[0][1:]) or  self.is_number(split_slash[0]):
    #             return True
    #     return False  
            
    # def contains_html_tags(self, text):
    #     # Define a regex pattern for HTML tags
    #     html_tag_pattern = re.compile(r'<[^>]+>')
    #     # Search for the pattern in the input text
    #     if html_tag_pattern.search(text):
    #         return True
    #     return False
    
    def buildDataset(
        self,
        is_entity=False,
        n=5
    ):
        """_summary_

        Args:
            n: denote the number of element by column and the the number of column

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
                    _file = pd.read_csv(f"{self.table_path}/{filed}")
                    total_column = _file.axes[1]
                    # get colums of each cleaned file csv         
                    df = _file.iloc[:, 1:n]
                    # take randomly n elements non NaN of each colummn
                    column_selected = []
                    i = 1
                    for col in df.columns:
                        non_nan_elements = df[col].dropna()
                        if len(non_nan_elements) < n:
                            # take all available element
                            selected_elements = non_nan_elements.tolist()
                        else:
                            # take n elements randomly
                            selected_elements = non_nan_elements.sample(n=n).tolist()
                        
                        # chek each element for coma and apply .split(',')
                        processed_elements = []
                        for elem in selected_elements:
                            if isinstance(elem, str):
                                # Split and choose one element randomly
                                split_elements = elem.split(',')
                                elem = random.choice(split_elements).strip()
                                if contains_html_tags(elem):
                                    continue
                                elif is_date(elem):
                                    continue
                                if not is_valid_string(elem) or 'http' in str(elem):
                                    # print(elem)
                                    continue
                                else:
                                    processed_elements.append(elem)
                            else:
                                processed_elements.append(elem)
                        if processed_elements:
                            processed_elements = str(processed_elements)
                            column_selected.append(processed_elements)
                            
                    if not column_selected:
                        if is_entity:
                            column_selected=[_file.columns[1]]
                        else:
                            column_selected = _file.iloc[0:2, 1:3].dropna().values.tolist()
                            # print(column_selected)
                            for elements in column_selected:
                                i = column_selected.index(elements) # save index of the Old value
                                for elt in elements:
                                    if contains_html_tags(elt):
                                        print(elt)
                                        elements.remove(elt)
                                        continue
                                    if "http" and "%" in elt and not contains_html_tags(elt):                                    
                                        new_element = decole_url_file(elt)
                                        index = elements.index(elt)
                                        # here we update the value the elements index
                                        elements[index] = new_element
                                    # update value of the column_select elements:
                                    column_selected[i] = elements
                                    
                        
                    max_returned_token = self.compute_max_token(len(str(column_selected)), 0)
                    while not max_returned_token:
                        print(column_selected)
                        if not isinstance(column_selected, str):
                            column_selected = str(column_selected)
                        column_selected = eval(column_selected)[-1]
                        if isinstance(column_selected, str) or isinstance(column_selected, int):
                            column_selected = [column_selected]
                        # print(column_selected)
                        prompt_input = str(column_selected)
                        # print(len(prompt_input))
                        max_returned_token = self.compute_max_token(len(prompt_input), 0)
                    if len(column_selected) == 1:
                        writer.writerow([filename, column_selected[0]])
                    else:
                        writer.writerow([filename, column_selected])
 
                else:
                    print("it is not csv file")
            csv_file.close()
        return self.output_dataset
    
    def _makeDataset(
        self,
        is_train=True,
        is_entity=False,
        split=0,
        n=5
    ):
        """_summary_

        Args:
            is_train (bool, optional): _description_. Defaults to True.
            is_entity (bool, optional): _description_. Defaults to False.
            n (int, optional): _description_. Defaults to 5.
        """
        _raw_dataset = self.buildDataset(n=n, is_entity=is_entity)
        if self.target_file_gt:
            with open(self.target_file_gt, 'r') as file1, open(_raw_dataset, 'r') as file2:
                _reader1 = csv.reader(file1)
                _reader2 = csv.reader(file2)
                
                csv1_data = [row for row in _reader1]
                csv2_data = [row for row in _reader2]     
                
                with open(self.output_dataset, 'w', newline='') as updated_file:
                    writer = csv.writer(updated_file)
                    writer.writerow(["tab_id", "columns", "entity"])
                    for row1 in csv1_data[split:]:
                        match_found = False
                        for row2 in csv2_data:
                            if row1[:1] == row2[:1]:
                                match_found = True
                                if is_train:
                                    row2.append(row1[1])
                                else:
                                    row2.append("NIL")
                                writer.writerow(row2)
                                # print(f"Row {row1} it is in CSV2")
                                break         
                        if match_found == False:
                            print(f"Row {row1} it is not in CSV2")
                    
        else:
            df = pd.read_csv(self.output_dataset)
            return df
        print("Comparison completed. Updated CSV2 saved as 'updated_csv2.csv'.")

    def _csv_to_jsonl(self, csv_path, json_path):
        """ 
            csv_path: path to csv file
            json_path: path to json file
        """
        df = self.openCSV(csv_path)
        datas = []
        
        for i in range(len(df['columns'])):
            # print(df['tab_id'][i])
            table = []
            try:         
                table = eval(df['columns'][i])
            except:
                table = [df['columns'][i]]
            uri = df['entity'][i]
            
            prompt_input = f"Please Give the  wikidata URI of of the topic of this table: {table}"
            if len(prompt_input) >= 2048:
                print(len(prompt_input))
            max_returned_token = self.compute_max_token(len(prompt_input), 0)
            
            while not max_returned_token:
                prompt_input = eval(prompt_input[61:])[-1]
                prompt_input = "Please Give the  wikidata URI of of the topic of this table: " + str(prompt_input)
                print(len(prompt_input))
                max_returned_token = self.compute_max_token(len(prompt_input), 0)

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
                            "content": f"""{{"table":  "{table}", "uri": "{uri}"}}"""
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
            
    
    def inference(self, model_id, user_input=None, temperature=0.00, frequency_penalty=0, presence_penalty=0, max_tokens=2048):
        """_summary_

        Args:
            model_id (_type_): _description_
            user_input (_type_, optional): _description_. Defaults to None.
            temperature (float, optional): _description_. Defaults to 0.00.
            frequency_penalty (int, optional): _description_. Defaults to 0.
            presence_penalty (int, optional): _description_. Defaults to 0.
            max_tokens (int, optional): _description_. Defaults to 2048.

        Returns:
            wikidata_uri: _description_: This function return the uri of wikidata correponding \nto the given table in input
        """
        chatBot = "Hi, I'm semantic annotation Agent. What can i do to help you today."
        if user_input is None:
            user_input = input('User: \n')
        # check the total length contex
        prompt_length = len(user_input)
        max_returned_token = self.compute_max_token(prompt_length=prompt_length, max_new_token=max_tokens)
        if not max_returned_token:
            user_input = eval(user_input[61:])[-1]
            user_input = str(user_input)
        conversation = [{"role": "user", "content": user_input}]
        
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
                    seed=44,
                    messages=message_input,
                    max_tokens=max_tokens
                )
            
                # Extract the chatbot response from API response
            chat_response = completion.choices[0].message.content    
            # Update conversation
            conversation.append({"role": "assistant", "content": chat_response})
            try:      
                result = json.loads(chat_response)
                # print(result)
                # result = eval(chat_response)
                label = result['table']
                uri = result['uri'] if 'uri' in result else ''
                
                print(f"The wikidata uri of the topic of this table is: {uri}")
            except json.JSONDecodeError as e:
                # Extraire la valeur de la clé "entities"\
                
                chat_response = chat_response.replace('"[', '[').replace(']"', ']')
                print(chat_response)
                # chat_response = eval(chat_response)
                # uri = chat_response['uri']
                uri = chat_response.split(f'"uri":')[-1].split('"')[1]
                print(f"The wikidata uri of the topic of this table is {uri}")
        except openai.InternalServerError as e:
            print(e)
            uri = ""
               
        return uri
            
    
    def _annotate(self, model, path=None, split=0):
        """_summary_

        Args:
            model (_type_): _description_: model id on openai
            path (_type_, optional): _description_: Defaults to None. Path of the csv dataset .
            split (int, optional): _description_: Defaults to 0.
        """
        if not path:
            filed = self.output_dataset
        else:
            filed = path
        header_cta = ["tab_id", "col_id", "annotation"]
        
        with open(self.target_file_to_annotate, "r") as csv_file:
            target_reader = csv.reader(csv_file)
            target_data = [row for row in target_reader]
            with open(self.file_annotated, 'w', newline='') as updated_file:
                writer = csv.writer(updated_file)
                # writer.writerow(header_cta)
                # check if it is csv file
                if filed.endswith(".csv"):
                    print(filed)
                    _file = pd.read_csv(f"{filed}") # open file with pandas
                    i = split
                    for data in target_data[split:]:
                        updated_cea_data = []   # at each iteration in reader_data, empty the list
                        table =  _file['columns'][i]
                        # get annotation of the column type
                        user_input = f"Please Give the  wikidata URI of of the topic of this table: {table}"                  
                        # Use connectionist approach
                        result = self.inference(model_id=model, user_input=user_input)                    
                        # add result of annation   
                        data.append(result)
                        updated_cea_data.append(data)
                            
                        #  write data in update cea file
                        writer.writerows(updated_cea_data)
                        print("*************************")
                        print(f"table {i} annotated")
                        print("*************************")
                        i += 1  
                    else:
                        print("it is not csv file")
