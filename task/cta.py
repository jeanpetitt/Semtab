import random
import pandas as pd
import os, json, csv
from .helper import getNameCsvFile
import re, openai
from .symbolic.api import openUrl, get_instance_of
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

class CTATask:
    
    def  __init__(
        self, 
        dataset_name = "cta",
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
        header=True,
    ):
        """_summary_

        Args:
            header (bool, optional):Defaults to True Set it to false if tables dataset don't have header.

        Returns:
            file: return a raw dataset
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
                    total_cols=len(_file.axes[1])
                    print("fichier:", filename, "Number of Column ", total_cols)
                    list_cell_selected = []
                    for col in _file.columns:
                        noNanElemenent = _file[col].values
                        noNanElemenent = [str(x) for x in noNanElemenent if (not isinstance(x, float) and not contains_html_tags(str(x)) and not 'http' in str(x))]
                        # print(noNanElemenent)
                        # noNanElemenent = ','.join(noNanElemenent)
                        # noNanElemenent = noNanElemenent.split(",")
                        
                        """ 
                            Each group of cells selected represent the elements that describe the best 
                            the column
                        """
                        if len(noNanElemenent) > 10:
                            _cell_selected = random.sample(noNanElemenent, k=6)
                            _cell_selected = [random.choice(x.split(',')) for x in _cell_selected]
                            # print(_cell_selected)
                            list_cell_selected.append(_cell_selected)
                        else:
                            _cell_selected = noNanElemenent
                            _cell_selected =  [random.choice(x.split(',')) for x in _cell_selected]
                            # print(_cell_selected)
                            list_cell_selected.append(_cell_selected)
                    col = 0
                    while col < total_cols:
                        writer.writerow([filename, col, list_cell_selected[col]])
                        col += 1
                else:
                    print("it is not csv file")
            csv_file.close()
        return self.output_dataset
    def _makeDataset(
        self,
        header=True,
        is_train=True
    ):
        """ 
            This function take two csv file which are almost same and compare the rows of the two files
            in order to create a new file that is same of the csv file 1
        """
        _raw_dataset, = self.buildDataset(
            header=header
        )
        # _raw_dataset, _target = self.raw_output_dataset, self.target_file
        if self.target_file_gt:
            with open(self.target_file_gt, 'r') as file1, open(_raw_dataset, 'r') as file2:
                _reader1 = csv.reader(file1)
                _reader2 = csv.reader(file2)
                
                csv1_data = [row for row in _reader1]
                csv2_data = [row for row in _reader2]     
                  
                with open(self.output_dataset, 'w', newline='') as updated_file:
                    writer.writerow(["tab_id", "col_id", "entities", "entity_type"]) 
                    for row1 in csv1_data:
                        match_found = False
                        for row2 in csv2_data:
                            updated_data = []
                            if row1[:2] == row2[:2]:
                                match_found = True
                                if is_train:
                                    row2.append(row1[2])
                                else: 
                                    row2.append("NIL")
                                updated_data.append(row2)
                                writer = csv.writer(updated_file)
                                writer.writerow(updated_data)
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
        
        for i in range(len(df['entities'])):
            list_entity = (df['entities'][i]).split("[]")
            list_entity = ','.join(list_entity)
            uri = df['entity_type'][i].split(',')

            if len(uri) == 1:
                ancestor = uri[0]
                descendant = uri[0]
                midlle = ''
            elif len(uri) == 2:
                ancestor = uri[0]
                descendant = uri[1]
                midlle = ''
            elif len(uri) > 2:
                ancestor = uri[0]
                descendant = uri[-1]
                midlle = uri[1:-1]
            datas.append(
                {
                    "messages":  [
                        {
                            "role": "system", 
                            "content": "Hi, I'm semantic annotation Agent. What can i do to help you today."
                        },
                        {
                            "role": "user", 
                            "content": f"Please Give the  wikidata URI of the entity type of the following entities: {list_entity}"
                        },      
                        {
                            "role": "assistant", 
                            "content": f"""{{"entities":  "{list_entity}", "AncestorType": "{ancestor}", "descendantType": "{descendant}", "middleTypes": "{midlle}"}}"""
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
            
    def compute_max_token(self, prompt_length, max_new_token):
        max_returned_tokens = max_new_token + prompt_length
        print("Prompt length:", prompt_length)
        assert max_returned_tokens <= self.context_length, (
            max_returned_tokens,
            self.context_length
        )
    
    def inference(self, model_id, user_input=None, temperature=0.80, frequency_penalty=0, presence_penalty=0, max_tokens=1024):
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
                # result = eval(chat_response)
                label = result['entities']
                
                if 'descendantType' in result:
                    uri = result['descendantType'].split(" ")[0]
                if 'AncestorType' in result:
                    uri = result['AncestorType'].split(" ")[0]
                else:
                    uri = ""
                
                print(f"The wikidata uri of the entity Type that match to these entity: {label} is {uri}")
            except json.JSONDecodeError as e:
                # Extraire la valeur de la clé "entities"\
                print(chat_response)


                # Extraire la valeur de la clé "descendantType"
                descendant_type_start_idx = chat_response.find('"descendantType":')
                descendant_type = ""
                if descendant_type_start_idx != -1:     
                    descendant_type_start_idx += len('"descendantType":')
                    descendant_type_end_idx = chat_response.find(',', descendant_type_start_idx)
                    if descendant_type_end_idx != -1:
                        descendant_type = chat_response[descendant_type_start_idx:descendant_type_end_idx].strip()
                        uri = descendant_type
                        uri = uri.strip('"')
                        uri = uri.split(" ")[0]
                    else:
                        uri = ""
                else:
                    uri = ""

                # Extraire la valeur de la clé "AncestorType"
                ancestor_type_start_idx = chat_response.find('"AncestorType":')
                ancestor_type = ""
                if ancestor_type_start_idx != -1:
                    ancestor_type_start_idx += len('"AncestorType":')
                    ancestor_type_end_idx = chat_response.find(',', ancestor_type_start_idx)
                #     if ancestor_type_end_idx != -1:            
                #         ancestor_type = chat_response[ancestor_type_start_idx:ancestor_type_end_idx].strip()
                #         uri = ancestor_type
                #         uri = uri.strip('"')
                #         uri = uri.split(" ")[0]
                #     else:
                #         uri = "NIL"
                # else:
                #     uri = "NIL"
                print(f"The wikidata uri of the entity Type that match to these entity is {uri}")
        except openai.InternalServerError as e:
            print(e)
            uri = ""
               
        return uri
    

    def correct_spelling(self, text):
        prompt = f"Don't argument in your answer. Correct the spelling of this text : \"{text}\""
        model = "gpt-4o"
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
        if len(corrected_text) == 1:
            corrected_text = "".join(corrected_text).strip()
        elif len(corrected_text) <=3:
            corrected_text = corrected_text[1]
            corrected_text = f"{corrected_text}.".strip(".").strip()
        else:
            corrected_text = corrected_text[3].split(".")[0].strip()
        print(f"the correct Text of {text} is {corrected_text}")
        return corrected_text
    
    def _annotate(self, model, path=None, split=0, is_combine_approach=False):
        """_summary_

        Args:
            model (_type_): _description_
            path (_type_, optional): _description_. Defaults to None.
            split (int, optional): _description_. Defaults to 0.
            is_combine_approach (bool, optional): _description_. Defaults to False.
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
                    # writer.writerow(header_cta)
                    # check if it is csv file
                    if dataset_path.endswith(".csv"):
                        print(dataset_path)
                        df = pd.read_csv(f"{dataset_path}") # open file with pandas
                        i = split
                        for data in target_data[split:]:
                            updated_cea_data = []   # at each iteration in reader_data, empty the list
                            list_entity =  df['entities'][i]     
                            # get annotation of the column type
                            user_input = f"Please Give the  wikidata URI of the enity type of the following entities: {list_entity}"                  
                            # Use symbolic approach combine with connectionist approach
                            if is_combine_approach:
                                list_entity = eval(list_entity)
                                uri_type_list = []
                                for label in list_entity[:random.randrange(1, 4)]:
                                    entity_id = openUrl(label)
                                    entity_id = entity_id.split('/')[-1] if entity_id else "nil"
                                    if entity_id.lower() != 'nil' and entity_id:
                                        uri_type = get_instance_of(entity_id)
                                        uri_type_list.append(uri_type)
                                    else:
                                        label = self.correct_spelling(label)
                                        entity_id = openUrl(label)
                                        entity_id = entity_id.split('/')[-1] if entity_id else "nil"
                                        if entity_id.lower() != 'nil' and entity_id:
                                            uri_type = get_instance_of(entity_id)
                                            uri_type_list.append(uri_type)
                                        
                                print(uri_type_list)
                                if uri_type_list:
                                    uri_type_list = [elem for elem in uri_type_list if elem is not None]
                                    result = get_most_common_element(uri_type_list)
                                    print("the wikidata uri of these entity is: ", result)
                                else:
                                    result = ''
                                
                                # use connectionist approach
                                if not result: 
                                    # get uri                     
                                    result = self.inference(model_id=model, user_input=user_input)
                            else:
                                result = self.inference(model_id=model, user_input=user_input)
                                if (result == '' or not result.split('Q')[-1].isdigit()) and result.lower() != "nil":
                                    list_entity = eval(list_entity)
                                    uri_type_list = []
                                    for label in list_entity[:3]:
                                        entity_id = openUrl(label)
                                        if entity_id:
                                            entity_id = entity_id.split('/')[-1]
                                            uri_type = get_instance_of(entity_id)
                                            uri_type_list.append(uri_type)
                                        else:
                                            label = self.correct_spelling(label)
                                            entity_id = openUrl(label)
                                            if entity_id:
                                                uri_type = get_instance_of(entity_id)
                                                uri_type_list.append(uri_type)
                                            else:
                                                uri_type_list.append(entity_id)
                                    print(uri_type_list)
                                    uri_type_list = [elem for elem in uri_type_list if elem is not None]
                                    result = get_most_common_element(uri_type_list)
                                    if not result:
                                        result = "NIL"                     
                                    print(result)
                                    
                            # add result of annation   
                            data.append(result)
                            updated_cea_data.append(data)
                                
                            #  write data in update cea file
                            writer.writerows(updated_cea_data)
                            print("*************************")
                            print(f"Col {i} annotated")
                            print("*************************")
                            i += 1
                        else:
                            print("it is not csv file")
        else:
            with open(self.file_annotated, 'w', newline='') as updated_file:
                writer = csv.writer(updated_file)
                # writer.writerow(header_cta)
                # check if it is csv file
                if dataset_path.endswith(".csv"):
                    print(dataset_path)
                    df = pd.read_csv(f"{dataset_path}") # open file with pandas
                    datas = df.values.tolist()
                    print(len(df))
                    i = split
                    for data in datas[split:]:
                        updated_cea_data = []   # at each iteration in reader_data, empty the list
                        list_entity =  df['entities'][i]     
                        # get annotation of the column type
                        user_input = f"Please Give the  wikidata URI of the enity type of the following entities: {list_entity}"                  
                        # Use symbolic approach combine with connectionist approach
                        if is_combine_approach:
                            list_entity = eval(list_entity)
                            uri_type_list = []
                            for label in list_entity[:random.randrange(1, 4)]:
                                entity_id = openUrl(label)
                                entity_id = entity_id.split('/')[-1] if entity_id else "nil"
                                if entity_id.lower() != 'nil' and entity_id:
                                    uri_type = get_instance_of(entity_id)
                                    uri_type_list.append(uri_type)
                                else:
                                    label = self.correct_spelling(label)
                                    entity_id = openUrl(label)
                                    entity_id = entity_id.split('/')[-1] if entity_id else "nil"
                                    if entity_id.lower() != 'nil' and entity_id:
                                        uri_type = get_instance_of(entity_id)
                                        uri_type_list.append(uri_type)                                        
                            print(uri_type_list)
                            if uri_type_list:
                                uri_type_list = [elem for elem in uri_type_list if elem is not None]
                                result = get_most_common_element(uri_type_list)
                                print("the wikidata uri of these entity is: ", result)
                            else:
                                result = ''                            
                            if not result: 
                                # get uri                     
                                result = self.inference(model_id=model, user_input=user_input)
                        # use connectionist approach
                        else:
                            result = self.inference(model_id=model, user_input=user_input)
                            if (result == '' or not result.split('Q')[-1].isdigit()) and result.lower() != "nil":
                                list_entity = eval(list_entity)
                                uri_type_list = []
                                for label in list_entity[:3]:
                                    entity_id = openUrl(label)
                                    if entity_id:
                                        entity_id = entity_id.split('/')[-1]
                                        uri_type = get_instance_of(entity_id)
                                        uri_type_list.append(uri_type)
                                    else:
                                        label = self.correct_spelling(label)
                                        entity_id = openUrl(label)
                                        if entity_id:
                                            uri_type = get_instance_of(entity_id)
                                            uri_type_list.append(uri_type)
                                        else:
                                            uri_type_list.append(entity_id)
                                print(uri_type_list)
                                uri_type_list = [elem for elem in uri_type_list if elem is not None]
                                result = get_most_common_element(uri_type_list)
                                if not result:
                                    result = "NIL"                     
                                print(result)                        
                        # add result of annation   
                        data.append(result)
                        updated_cea_data.append(data)
                            
                        #  write data in update cea file
                        writer.writerows(updated_cea_data)
                        print("*************************")
                        print(f"Col {i} annotated")
                        print("*************************")
                        i += 1
                else:
                    print("it is not csv file")
