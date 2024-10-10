from editdistance import eval as eval_distance
import requests
from task.symbolic.api import check_entity_properties_cea

def openUrl(query):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "language": "en",
        "format": "json",
        "search": query,
        "limit": 99,

    }
    uri = ""
    output = []
    try:
        response = requests.get(url, params=params)
        if "search" in response.json():
            result = response.json()["search"]
            if len(result) <= 5:
                best_result = sorted(result, key=lambda r: eval_distance(query, r['label']))
                result = f"http://www.wikidata.org/entity/{best_result[0]['id']}"
                output.append(
                    f" - Input : {query} \n\t-> id = {best_result[0]['id']}, link ={result} ")
                print(output)
                return [result]
            elif len(result) >5:
                best_result = result[:30]
                return best_result
    except:
        print("Internet connection error")
        return None
  
import openai, os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ['OPENAI_API_KEY']
openai.api_key = api_key
def correct_spelling(text):
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
        if len(corrected_text) == 1:
            corrected_text = "".join(corrected_text).strip()
        elif len(corrected_text) <=3:
            corrected_text = corrected_text[1]
            corrected_text = f"{corrected_text}.".strip(".").strip()
        else:
            corrected_text = corrected_text[3].split(".")[0].strip()
        print(corrected_text)
        return corrected_text
    
import csv
import requests

def get_wikidata_label(wikidata_id):
    """
    Get the label of a Wikidata entity by its QID.
    """
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if wikidata_id not in data['entities']:
            return None
        entity = data['entities'][wikidata_id]
        return entity['labels']['en']['value'] if 'en' in entity['labels'] else None
    return None

def get_wikidata_properties(wikidata_id):
    """
    Get properties from a Wikidata entity.
    :param wikidata_id: Wikidata ID (Qxxx)
    :return: Dictionary of properties and their QID values
    """
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if not wikidata_id in data['entities']:
            return {}
        entity = data['entities'][wikidata_id]
        claims = entity['claims']
        properties = {}
        
        for prop, val in claims.items():
            for claim in val:
                if 'mainsnak' in claim and 'datavalue' in claim['mainsnak']:
                    data_value = claim['mainsnak']['datavalue']
                    if data_value['type'] == 'wikibase-entityid':
                        # print(data_value)
                        if "numeric-id" not in data_value['value']:
                            continue
                        qid = f"Q{data_value['value']['numeric-id']}"
                        properties[qid] = qid
        # print(properties)
        return properties
    return {}

def is_edit_distance_one(word1, word2):
    # If the length difference is greater than 1, the edit distance can't be 1
    if abs(len(word1) - len(word2)) > 1:
        return False

    # Initialize variables
    len1, len2 = len(word1), len(word2)
    i = j = 0
    edit_count = 0

    while i < len1 and j < len2:
        if word1[i] != word2[j]:
            # If there's more than one edit, return False
            if edit_count == 1:
                return False
            edit_count += 1

            # If the strings are of the same length, increment both pointers
            if len1 == len2:
                i += 1
                j += 1
            # If word1 is longer, increment the pointer for word1
            elif len1 > len2:
                i += 1
            # If word2 is longer, increment the pointer for word2
            else:
                j += 1
        else:
            i += 1
            j += 1

    # If there's one more character left in either string
    if i < len1 or j < len2:
        edit_count += 1

    return edit_count == 1

previous_state = None
current_state = None
# Function to update the state and check for changes
def state_has_changed(new_state):
    global previous_state
    global current_state
    current_state = new_state
    state_changed = False

    if previous_state is not None and current_state != previous_state:
        state_changed = True
    
    # Update the previous state for the next check
    previous_state = current_state

    return state_changed

def action(current_properties, label, data_reader, row, previous_row, header, i, output_file):
    for prop, qid in current_properties.items():
        wikidata_label = get_wikidata_label(qid)
        if not wikidata_label:
            continue
        if wikidata_label.lower() == label.lower():
            data_reader[i][5] = f"http://www.wikidata.org/entity/{qid}"
            row[5] = f"http://www.wikidata.org/entity/{qid}"
            break
        elif is_edit_distance_one(wikidata_label.lower(), label.lower().strip()):
            data_reader[i][5] = f"http://www.wikidata.org/entity/{qid}"
            row[5] = f"http://www.wikidata.org/entity/{qid}"  # Assign the correct Wikidata ID
            break
    if previous_row:
        if int(previous_row[2]) == 0:
            data_reader[i-1][5] = id
            print(data_reader[i-1][5])
            with open(output_file, mode='w') as output:
                writer = csv.writer(output)
                writer.writerow(header)
                writer.writerows(data_reader[:i])
                output.close()
                
                
        
        

def process_file(input_file, output_file, split=0):
    """
    Process the CSV file and update Wikidata IDs.
    :param input_file: Path to the input CSV file
    :param output_file: Path to the output CSV file with updated Wikidata IDs
    """
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        data_reader = [data for data in reader]
        with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)
            outfile.close()

        current_properties = {}
        i = split
        previous_row = None
        for row in data_reader[split:]:
            col_id = int(row[2])  # Third column is crucial (index 2)
            label = row[3]  # Fourth column is the label (index 3)
            # identifier = None          
            
            if col_id == 0:
                identifier = str(row[5]).split("/")[-1]  # Sixth column is the Wikidata ID (index 5)
                row[5] = f"http://www.wikidata.org/entity/{identifier}"
                # current_properties = get_wikidata_properties(current_wikidata_id)
                # print(col_id, identifier)
            else:
                if not state_has_changed(identifier) :  
                    # print(col_id, identifier)
                    current_properties = get_wikidata_properties(identifier)
                    if not current_properties:
                        context = eval(row[4])
                        entity_ids = openUrl(str(context[0]).strip(".").strip())                       
                        # print(id)
                        if entity_ids:
                            if len(entity_ids) == 1:
                                id = ''.join(entity_ids)
                                current_properties = get_wikidata_properties(identifier)
                                action(
                                    current_properties=current_properties,
                                    label=label,
                                    data_reader=data_reader,
                                    row=row,
                                    previous_row=previous_row,
                                    header=header,
                                    i=i,
                                    output_file=output_file
                                )
                            else:
                                id = check_entity_properties_cea(
                                    entity_ids=entity_ids,
                                    property_values=context,
                                    label_current=label
                                )
                                if id:
                                    identifier = id
                                    current_properties = get_wikidata_properties(identifier)
                                    action(
                                        current_properties=current_properties,
                                        label=label,
                                        data_reader=data_reader,
                                        row=row,
                                        previous_row=previous_row,
                                        header=header,
                                        i=i,
                                        output_file=output_file
                                    )
                                    
                        else:
                            if previous_row:
                                if int(previous_row[2]) == 0:
                                    label_identifier = correct_spelling(context[0])
                                    entity_ids = openUrl(label_identifier)
                                    if len(entity_ids) == 1:
                                        if id:
                                            identifier = "".join(id).split("/")[-1]
                                            current_properties = get_wikidata_properties(identifier)
                                            action(
                                                current_properties=current_properties,
                                                label=label,
                                                data_reader=data_reader,
                                                row=row,
                                                previous_row=previous_row,
                                                header=header,
                                                i=i,
                                                output_file=output_file
                                            )
                                        else:
                                            if previous_row:
                                                if int(previous_row[2]) == 0:
                                                    label_identifier = context[0].split("(")[0].strip()
                                                    id = openUrl(label_identifier)
                                                    if id:
                                                        identifier = "".join(id).split("/")[-1]
                                                        current_properties = get_wikidata_properties(identifier)
                                                        action(
                                                            current_properties=current_properties,
                                                            label=label,
                                                            data_reader=data_reader,
                                                            row=row,
                                                            previous_row=previous_row,
                                                            header=header,
                                                            i=i,
                                                            output_file=output_file
                                                        ) 
                                    else:
                                        id = check_entity_properties_cea(
                                            entity_ids=entity_ids,
                                            property_values=context,
                                            label_current=label
                                        )
                                        if id:
                                            identifier = id
                                            current_properties = get_wikidata_properties(identifier)
                                            action(
                                                current_properties=current_properties,
                                                label=label,
                                                data_reader=data_reader,
                                                row=row,
                                                previous_row=previous_row,
                                                header=header,
                                                i=i,
                                                output_file=output_file
                                            )                                                                                                
                    else:
                        found = False
                        for prop, qid in current_properties.items():
                            # print(current_properties)
                            wikidata_label = get_wikidata_label(qid)
                            # print(len(current_properties.items()), wikidata_label, label)
                            if not wikidata_label:
                                continue
                            if wikidata_label.lower() == label.lower():
                                found = True
                                data_reader[i][5] = f"http://www.wikidata.org/entity/{qid}"
                                row[5] = f"http://www.wikidata.org/entity/{qid}"
                                break
                            elif is_edit_distance_one(wikidata_label.lower(), label.lower().strip()):
                                found = True
                                data_reader[i][5] = f"http://www.wikidata.org/entity/{qid}"
                                row[5] = f"http://www.wikidata.org/entity/{qid}"  # Assign the correct Wikidata ID
                                break
                        # if entity does not found, that means that the identifier was not the correct identifier
                        
                        if not found:
                            context = eval(row[4])
                            entity_ids = openUrl(str(context[0]).strip(".").strip())                       
                            # print(id)
                            if entity_ids:
                                if len(entity_ids) == 1:
                                    id = ''.join(entity_ids)
                                    identifier = id
                                    current_properties = get_wikidata_properties(identifier)
                                    action(
                                        current_properties=current_properties,
                                        label=label,
                                        data_reader=data_reader,
                                        row=row,
                                        previous_row=previous_row,
                                        header=header,
                                        i=i,
                                        output_file=output_file
                                    )
                                else:
                                    id = check_entity_properties_cea(
                                        entity_ids=entity_ids,
                                        property_values=context,
                                        label_current=label
                                    )
                                    if id:
                                        identifier = id
                                        current_properties = get_wikidata_properties(identifier)
                                        action(
                                            current_properties=current_properties,
                                            label=label,
                                            data_reader=data_reader,
                                            row=row,
                                            previous_row=previous_row,
                                            header=header,
                                            i=i,
                                            output_file=output_file
                                        )
                                        
                            else:
                                if previous_row:
                                    if int(previous_row[2]) == 0:
                                        label_identifier = correct_spelling(context[0])
                                        entity_ids = openUrl(label_identifier)
                                        if entity_ids:                                           
                                            if len(entity_ids) == 1:
                                                id = ''.join(entity_ids)
                                                identifier = id
                                                current_properties = get_wikidata_properties(identifier)
                                                action(
                                                    current_properties=current_properties,
                                                    label=label,
                                                    data_reader=data_reader,
                                                    row=row,
                                                    previous_row=previous_row,
                                                    header=header,
                                                    i=i,
                                                    output_file=output_file
                                                )
                                            else:
                                                id = check_entity_properties_cea(
                                                    entity_ids=entity_ids,
                                                    property_values=context,
                                                    label_current=label
                                                )
                                                if id:
                                                    identifier = id
                                                    current_properties = get_wikidata_properties(identifier)
                                                    action(
                                                        current_properties=current_properties,
                                                        label=label,
                                                        data_reader=data_reader,
                                                        row=row,
                                                        previous_row=previous_row,
                                                        header=header,
                                                        i=i,
                                                        output_file=output_file
                                                    )
                                                
                                        else:
                                            if previous_row:
                                                if int(previous_row[2]) == 0:
                                                    label_identifier = context[0].split("(")[0].strip()
                                                    id = openUrl(label_identifier)
                                                    if id:
                                                        if len(id) == 1:
                                                            id = ''.join(id)
                                                            identifier = id
                                                            current_properties = get_wikidata_properties(identifier)
                                                            action(
                                                                current_properties=current_properties,
                                                                label=label,
                                                                data_reader=data_reader,
                                                                row=row,
                                                                previous_row=previous_row,
                                                                header=header,
                                                                i=i,
                                                                output_file=output_file
                                                            )
                                                        else:
                                                            id = check_entity_properties_cea(
                                                                entity_ids=entity_ids,
                                                                property_values=context,
                                                                label_current=label
                                                            )
                                                            if id:
                                                                identifier = id
                                                                current_properties = get_wikidata_properties(identifier)
                                                                action(
                                                                    current_properties=current_properties,
                                                                    label=label,
                                                                    data_reader=data_reader,
                                                                    row=row,
                                                                    previous_row=previous_row,
                                                                    header=header,
                                                                    i=i,
                                                                    output_file=output_file
                                                                )
                    
                else:
                    current_properties = get_wikidata_properties(identifier)
                    if not current_properties:
                        context = eval(row[4])
                        id = openUrl(str(context[0]).strip(".").strip())                             
                        # print(id)
                        if id:
                            if len(id) == 1:
                                id = ''.join(id)
                                identifier = id
                                current_properties = get_wikidata_properties(identifier)
                                action(
                                    current_properties=current_properties,
                                    label=label,
                                    data_reader=data_reader,
                                    row=row,
                                    previous_row=previous_row,
                                    header=header,
                                    i=i,
                                    output_file=output_file
                                )
                            else:
                                id = check_entity_properties_cea(
                                    entity_ids=entity_ids,
                                    property_values=context,
                                    label_current=label
                                )
                                if id:
                                    identifier = id
                                    current_properties = get_wikidata_properties(identifier)
                                    action(
                                        current_properties=current_properties,
                                        label=label,
                                        data_reader=data_reader,
                                        row=row,
                                        previous_row=previous_row,
                                        header=header,
                                        i=i,
                                        output_file=output_file
                                    )
                        else:
                            label_identifier = correct_spelling(context[0])
                            id = openUrl(label_identifier)
                            if id:
                                if len(id) == 1:
                                    id = ''.join(id)
                                    identifier = id
                                    current_properties = get_wikidata_properties(identifier)
                                    action(
                                        current_properties=current_properties,
                                        label=label,
                                        data_reader=data_reader,
                                        row=row,
                                        previous_row=previous_row,
                                        header=header,
                                        i=i,
                                        output_file=output_file
                                    )
                                else:
                                    id = check_entity_properties_cea(
                                        entity_ids=entity_ids,
                                        property_values=context,
                                        label_current=label
                                    )
                                    if id:
                                        identifier = id
                                        current_properties = get_wikidata_properties(identifier)
                                        action(
                                            current_properties=current_properties,
                                            label=label,
                                            data_reader=data_reader,
                                            row=row,
                                            previous_row=previous_row,
                                            header=header,
                                            i=i,
                                            output_file=output_file
                                        )
                            else:
                                if previous_row:
                                    if int(previous_row[2]) == 0:
                                        label_identifier = context[0].split("(")[0].strip()
                                        id = openUrl(label_identifier)
                                        if id:
                                            if len(id) == 1:
                                                id = ''.join(id)
                                                identifier = id
                                                current_properties = get_wikidata_properties(identifier)
                                                action(
                                                    current_properties=current_properties,
                                                    label=label,
                                                    data_reader=data_reader,
                                                    row=row,
                                                    previous_row=previous_row,
                                                    header=header,
                                                    i=i,
                                                    output_file=output_file
                                                )
                                            else:
                                                id = check_entity_properties_cea(
                                                    entity_ids=entity_ids,
                                                    property_values=context,
                                                    label_current=label
                                                )
                                                if id:
                                                    identifier = id
                                                    current_properties = get_wikidata_properties(identifier)
                                                    action(
                                                        current_properties=current_properties,
                                                        label=label,
                                                        data_reader=data_reader,
                                                        row=row,
                                                        previous_row=previous_row,
                                                        header=header,
                                                        i=i,
                                                        output_file=output_file
                                                    )
                                                    
                                                               
            print(f"=======================================================")
            print(f"cell {i}, label={label}, id={row[5]}, identifier=http://www.wikidata.org/entity/{identifier}")
            print(f"=======================================================")
            if col_id == 0:
                qid = str(row[5]).split("/")[-1].strip()
                row[5] = row[5] = f"http://www.wikidata.org/entity/{qid}"
            with open(output_file, mode='a', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(row)
                outfile.close()
            previous_row = row
            i += 1
        # writer.writerows(data_reader)

process_file('dataset/llm_test_cea_dataset.csv', 'output6.csv', split=74623)