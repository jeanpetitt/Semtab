import math
import requests
from dotenv import load_dotenv
import os
from editdistance import eval as eval_distance
import json
import re
from urllib.parse import unquote
from .utils import *

from SPARQLWrapper import SPARQLWrapper, JSON



load_dotenv()
def get_instance_of(entity_id):
    base_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "format": "json",
        "props": "claims"
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    instance_of_property_id = "P31"  # ID of the property "instance of"

    # Check answer
    if "entities" in data and entity_id in data["entities"]:
        entity = data["entities"][entity_id]
        if "claims" in entity and instance_of_property_id in entity["claims"]:
            instance_claims = entity["claims"][instance_of_property_id]
            if len(instance_claims) > 0 and "mainsnak" in instance_claims[0] and "datavalue" in instance_claims[0]["mainsnak"]:
                instance_value = instance_claims[0]["mainsnak"]["datavalue"]["value"]
                if "id" in instance_value:
                    return instance_value["id"]
        elif "claims" in entity and instance_of_property_id in entity["claims"]:
            instance_claims = entity["claims"][instance_of_property_id]
            if len(instance_claims) > 0 and "mainsnak" in instance_claims[0] and "datavalue" in instance_claims[0]["mainsnak"]:
                instance_value = instance_claims[0]["mainsnak"]["datavalue"]["value"]
                if "id" in instance_value:
                    return instance_value["id"]

    return None

def openUrl(query, context=None, is_horizontal=True):
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
    best_result = []
  
    print(query)
    # try:
    response = requests.get(url, params=params)
    if "search" in response.json():
                result = response.json()["search"]
                if is_horizontal:           
                    if len(result) == 1:
                        if context:
                            uri = [f"{result[0]['id']}"]
                            output.append(
                            f" - Input : {query} \n\t-> id = {result[0]['id']}, link ={uri} ")
                            print(output)
                            return uri
                        else:
                            uri = f"http://www.wikidata.org/entity/{result[0]['id']}"
                        output.append(
                        f" - Input : {query} \n\t-> id = {result[0]['id']}, link ={uri} ")
                        print(output)
                        return uri
                    elif len(result) >= 2:
                        if context:
                            uri = []
                            best_result = sorted(result, key=lambda r: eval_distance(query, r['label']))
                            if best_result:
                                for result in best_result:
                                    uri.append(result['id'])
                            return uri
                        else:
                            best_result = sorted(result, key=lambda r: eval_distance(query, r['label']))[0]   
                            if best_result:
                                uri = f"http://www.wikidata.org/entity/{best_result['id']}"
                                output.append(
                                f" - Input : {query} \n\t-> id = {best_result['id']}, label = {best_result['label']}, link ={uri} ")
                                print(output)
                                return uri
                            else:
                                output.append(f" - Input : {query} \n\t-> ----Not Found----")
                                print(output)
                                return None
                    else:
                        if context:
                            uri = []
                            return uri
                        else:
                            return None
                else:
                    if len(result) >= 1 and len(result) <=2:
                        if context:
                            uri = []
                            for res in result:
                                uri.append(res['id'])
                            return uri
                        else:
                            uri = f"http://www.wikidata.org/entity/{result[0]['id']}"
                        output.append(
                        f" - Input : {query} \n\t-> id = {result[0]['id']}, link ={uri} ")
                        print(output)
                        return uri
                    elif len(result) > 2:
                        if context:
                            uri = []
                            best_result = sorted(result, key=lambda r: eval_distance(query, r['label']))
                            if best_result:
                                for result in best_result:
                                    uri.append(result['id'])
                            return uri
                        else:
                            best_result = sorted(result, key=lambda r: eval_distance(query, r['label']))[0]   
                            if best_result:
                                uri = f"http://www.wikidata.org/entity/{best_result['id']}"
                                output.append(
                                f" - Input : {query} \n\t-> id = {best_result['id']}, label = {best_result['label']}, link ={uri} ")
                                print(output)
                                return uri
                            else:
                                output.append(f" - Input : {query} \n\t-> ----Not Found----")
                                print(output)
                                return None
                    else:
                        if context:
                            return []
                        else:
                            return None
    else:
                if context:
                    return []
                else:
                    return None

    # except:
    #     print("Internet connection error")
    #     if context:
    #         return []
    #     else:
    #         return None
        
def get_entity_with_id(entity_id):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "language": "en",
        "format": "json",
        "search": entity_id

    }
    try:
            response = requests.get(url, params=params)
            # if response.json()["search"] != []:
            result = response.json()["search"]
            # print(result)
            if len(result) >= 1:
                result = result[0]  

    except ValueError as e:
        print("Internet connection error")

    return result

# check if string is a numer 
def is_number(string):
    return re.match(r"^[+-]?\d+(\.\d+)?$", str(string)) is not None

# this function check if a list is a sublist of annather list
def is_subpart(subpart: list, mainpart: list):
    new_subpart = []
    print(subpart)
    for sub in subpart:
        if sub in mainpart:
            new_subpart.append(sub)
        elif str(sub).startswith('+') or str(sub).startswith('-'):
            # mainpart = [item for item in mainpart if isinstance(item, (int, float)) or (isinstance(item, str) and is_number(item))]
            for val in mainpart:
                if val == FlexibleValue(sub):
                    print(val, sub)
                    index = subpart.index(sub)
                    subpart[index] = val
                else:
                    if is_number(val) and is_number(sub):
                        positif = "".join(val.split("+")[-1].split("."))
                        negatif = "".join(val.split("-")[-1].split("."))
                        if "+" in val and len(positif) < 6:
                            val1 = float(val.split("+")[-1])
                            val2 = float(str(sub).split("+")[-1])
                            dif =  val1 - val2
                            if dif < 0:
                                dif = -(dif)
                            if dif > 0 and dif < 2.89:
                                index = subpart.index(sub)
                                subpart[index] = val
                        elif "-" in  val and len(negatif) < 6:
                            val1 = float(val.split("-")[-1])
                            val2 = float(str(sub).split("-")[-1])
                            if dif < 0:
                                dif = -(dif)
                            if dif > 0 and dif < 2.89:
                                index = subpart.index(sub)
                                subpart[index] = val
                        else:
                            val1 = float(val[1:3])
                            val2 = float(str(sub)[1:3])
                            if dif < 0:
                                dif = -(dif)
                            if dif > 0 and dif <= 11.5:
                                index = subpart.index(sub)
                                subpart[index] = val         
                    else:
                        continue
        else:
            continue
 
    return subpart

# check if number is betwen two number [a, b]
def is_inInterval(number, L: list):
    if len(L) < 2 or len(L) > 2:
        raise ValueError("The interval is a list with 2 as a length. example [1, 2]")
    else:
        if L[0] < L[1]:
            if number >= L[0] and number <= L[1]:
                return True
        else:
            if number >= L[1] and number <= L[0]:
                return True
    return False

# check if object have string value
def is_date(string):
    split_slash = len(str(string.split("/")))
    split_tiret = len(str(string.split("-")))
    if split_slash == 3 or split_tiret == 3:
        if is_number(split_tiret[0]) or is_number(split_tiret[0][1:]) or is_number(split_slash[0]):
            return True
        return False  
    
def have_string(object):
    if isinstance(object, list):
        for item in object:
            if isinstance(item, str):
                if item.isdigit() or is_date(item) or is_number(item):
                    continue
                return True
    else:
        if str(object).isdigit() or is_number(str(object) or is_date(str(object))):
            return False
        else:
            return True
    return False

# extract number in a string
def extract_numbers(s):
    # Use regular expression to fin all number in a string
    numbers = re.findall(r'-?\d+\.\d+|-?\d+', s)
    # convert result into flottant
    return [float(num) for num in numbers]

# check if two words or sentence can be consider as equivalent
def is_equal_distance(str1, str2):
    distance = eval_distance(str1, str2)
    if distance <=3:
        return True
    return False  

# get candidate property for the cpa task
def check_entity_properties_cpa(entity_ids, property_values, is_horizontal=False, label_value=None, is_combined=False, index=None, is_ceaTask=False):
    if is_horizontal:
        context_have_string_value = have_string(label_value)
    else:       
        context_have_string_value = have_string(property_values)
    for entity_id in entity_ids:
        url = f"https://www.wikidata.org/w/api.php?action=wbgetclaims&entity={entity_id}&format=json"
        response = requests.get(url)
        data = response.json()

        if "claims" in data:
            claims = data["claims"]
            entity_property_values = []
            property_value_ids = []
            entity_properties = []

            for property_id, property_claims in claims.items():
                i = 1
                for claim in property_claims:
                    if "mainsnak" in claim and "datavalue" in claim["mainsnak"]:
                        value = claim["mainsnak"]["datavalue"]["value"]
                        if is_horizontal and not is_combined or not is_horizontal:
                            if 'http' in label_value and 'http' in value:
                                _value1 = "".join(label_value.split("/")[:-1])
                                _value2 = "".join(value.split("/")[:-1])
                                if _value1 == _value2:
                                    print(f"http://www.wikidata.org/entity/{entity_id}")
                                    if is_ceaTask: return f"http://www.wikidata.org/entity/{entity_id}"
                                    return {
                                        'prop_id': claim["mainsnak"]["property"],
                                        'prop_value': label_value
                                    }
                            # check if is media
                            elif (claim["mainsnak"]["property"] == "P10" or "P2716" or claim["mainsnak"]["property"] == "P18") and  "%" in label_value:
                                filename = label_value.split('/')[-1]
                                decoded_filename = unquote(filename)
                                if value == decoded_filename:
                                    print(f"http://www.wikidata.org/entity/{entity_id}")
                                    if is_ceaTask: return f"http://www.wikidata.org/entity/{entity_id}"
                                    return {
                                        'prop_id': claim["mainsnak"]["property"],
                                        'prop_value': label_value
                                    }
                                    
                        if is_horizontal and not is_combined:
                            if value == FlexibleValue(label_value):
                                print(f"http://www.wikidata.org/entity/{entity_id}")
                                return {
                                    'prop_id': property_id,
                                    'prop_value': label_value
                                }
                        elif value == FlexibleValue(label_value):
                            print(f"http://www.wikidata.org/entity/{entity_id}")
                            if is_ceaTask: return f"http://www.wikidata.org/entity/{entity_id}"
                            return {
                                'prop_id': property_id,
                                'prop_value': label_value
                            }
                        if context_have_string_value:
                            if 'id' in value and not isinstance(value, str):
                                # print(value['id'])
                                result = get_entity_with_id(value['id'])
                                label = result['label'] if 'label' in result else ''
                                idx = result['id'] if 'id' in result else ''
                                if is_horizontal and not is_combined:
                                    if ',' in label:
                                        label = [lab.strip() for lab in label.split(",")]
                                        if label_value in label:
                                            return {
                                                'prop_id': property_id,
                                                'prop_value': label_value
                                            } 
                                    if label == label_value or idx == label_value:
                                        return {
                                            'prop_id': property_id,
                                            'prop_value': label_value
                                        }
                                else:
                                    # should be deleted after the challenge
                                    if label == label_value or idx == label_value:
                                        print(f"http://www.wikidata.org/entity/{entity_id}")
                                        if is_ceaTask: return f"http://www.wikidata.org/entity/{entity_id}"
                                        return {
                                            'prop_id': property_id,
                                            'prop_value': label_value
                                        }
                                    if label in property_values:
                                        entity_property_values.append(label)
                                    else:
                                        continue
                                    if i == len(property_values):
                                        break
                                    else:
                                        i += 1
                                        continue
                        if 'amount' in value and not isinstance(value, str) and is_number(label_value):
                            if is_horizontal and not is_combined:
                                if value['amount'] == FlexibleValue(label_value):
                                    if is_ceaTask: return f"http://www.wikidata.org/entity/{entity_id}"
                                    return {
                                        'prop_id': property_id,
                                        'prop_value': label_value
                                    }
                            else:
                                value = value['amount']
                                # comput the difference between two number due to update
                                """ At the end of the challenfe i should reformat this code"""
                                if FlexibleValue(value) == label_value or value == label_value:
                                    if is_ceaTask: return f"http://www.wikidata.org/entity/{entity_id}"
                                    return {
                                        'prop_id': property_id,
                                        'prop_value': label_value
                                    }
                                elif "".join( label_value.split("+")[-1].split(".")).isdigit() or "".join( label_value.split("-")[-1].split(".")).isdigit():
                                    # print(value)
                                    positif = "".join( label_value.split("+")[-1].split("."))
                                    negatif = "".join( label_value.split("-")[-1].split("."))
                                    if "+" in value and len(positif) < 5:
                                        val1 = float(value.split("+")[-1])
                                        val2 = float(label_value.split("+")[-1])
                                    elif "-" in value and len(negatif) < 5:
                                        val1 = float(value.split("-")[-1])
                                        val2 = float(label_value.split("-")[-1])
                                    else:
                                        val1 = float(value[1:3])
                                        val2 = float(label_value[:2])
                                        # print(f"{val1, val2}")
                                    dif =  val1 - val2
                                    if dif < 0:
                                        dif = -(dif)
                                    # print(f"{val1, val2, dif}")
                                    # print(dif)
                                    if dif > 0 and dif < 2.89:
                                        if is_ceaTask: return f"http://www.wikidata.org/entity/{entity_id}"
                                        return {
                                            'prop_id': property_id,
                                            'prop_value': label_value
                                        }
                                    else:   
                                        # the rest of this code should be deleted                     
                                        entity_property_values.append(value)
                        elif 'text' in value and not isinstance(value, str):
                            value = value['text']             
                            if is_horizontal and not is_combined:
                                if value == label_value or is_equal_distance(label_value, value):
                                    print(f"http://www.wikidata.org/entity/{entity_id}")
                                    return {
                                        'prop_id': property_id,
                                        'prop_value': label_value
                                    }
                            else:
                                if FlexibleValue(value) == label_value or is_equal_distance(label_value, value):
                                    if is_ceaTask: return f"http://www.wikidata.org/entity/{entity_id}"
                                    return {
                                        'prop_id': property_id,
                                        'prop_value': label_value
                                    }
                                else: 
                                    # the rest of this code should be deleted                       
                                    entity_property_values.append(value)
                        elif 'longitude' and 'latitude' in value and not isinstance(value, str):
                            precision = value['precision']
                            if precision and  precision > 0:
                                longitude = round(float(value['longitude']), int(-math.log10(precision)))
                                latitude = round((value['latitude']), int(-math.log10(precision)))
                            else:
                                longitude = value['longitude']
                                latitude = value['latitude']
                            point_value = Point(longitude, latitude)
                            value = f"Point({longitude}, {latitude})"
                            # Extract x=longitude and y=latitude in Point(x,y)
                            # horizontal case
                            if is_horizontal and not is_combined and "point(" in str(label_value).lower():
                                point_value_label = extract_numbers(label_value)
                                point_value_label = Point(point_value_label[0], point_value_label[1])
                                if point_value == point_value_label:
                                    print(f"http://www.wikidata.org/entity/{entity_id}")
                                    return {
                                        'prop_id': property_id,
                                        'prop_value': label_value
                                    }
                            else:
                                if "point(" in str(label_value).lower():
                                    point_value_label = extract_numbers(label_value)
                                    point_value_label = Point(point_value_label[0], point_value_label[1])
                                    if point_value == point_value_label:
                                        if is_ceaTask: return f"http://www.wikidata.org/entity/{entity_id}"
                                        return {
                                            'prop_id': property_id,
                                            'prop_value': label_value
                                        }
                                else:
                                    # the rest of this code should be deleted
                                    entity_property_values.append(value)
                        elif 'time' in value and not isinstance(value, str):
                            value = value['time']
                            if is_horizontal and not is_combined:
                                if label_value[0:10] == value[1:11] or label_value[0:10] == value[0:10] or label_value[0:4] == value[0:4]  or label_value[0:4] == value[1:5] or label_value[0:7] == value[:7]or label_value[0:7] == value[1:8]:
                                    print(f"http://www.wikidata.org/entity/{entity_id}")
                                    return {
                                        'prop_id': property_id,
                                        'prop_value': label_value
                                    }
                            else:
                                if label_value[0:10] == value[1:11] or label_value[0:10] == value[0:10] or label_value[0:4] == value[0:4]  or label_value[0:4] == value[1:5] or label_value[0:7] == value[:7]or label_value[0:7] == value[1:8]:
                                    if is_ceaTask: return f"http://www.wikidata.org/entity/{entity_id}"
                                    return {
                                        'prop_id': property_id,
                                        'prop_value': label_value
                                    }
                                # the rest of this code should be deleted
                                entity_properties.append(value)
                            if is_horizontal and not is_combined:
                                if value == label_value:
                                    print(f"http://www.wikidata.org/entity/{entity_id}")
                                    return {
                                        'prop_id': property_id,
                                        'prop_value': label_value
                                    }
                            else:
                                if value == label_value:
                                    if is_ceaTask: return f"http://www.wikidata.org/entity/{entity_id}"
                                    return {
                                        'prop_id': property_id,
                                        'prop_value': label_value
                                    }
                                # the rest of this code should be deleted
                                entity_property_values.append(value)
                        else:
                            # print(value)
                            entity_property_values.append(value)
                        if property_id == "P571":
                            if is_horizontal and not is_combined:
                                if value == label_value:
                                    if is_ceaTask: return entity_id
                                    print(f"http://www.wikidata.org/entity/{entity_id}")
                                    return {
                                        'prop_id': property_id,
                                        'prop_value': label_value
                                    }
                            else:  
                                if value == label_value:
                                    if is_ceaTask: return f"http://www.wikidata.org/entity/{entity_id}"
                                    return {
                                        'prop_id': property_id,
                                        'prop_value': label_value
                                    } 
                                # the rest of this code should be deleted
                                if value in property_values:
                                    pass
                                elif value.isdigit():
                                    if is_horizontal and not is_combined:
                                        if value == label_value[:4]:
                                            print(f"http://www.wikidata.org/entity/{entity_id}")
                                            return {
                                                'prop_id': property_id,
                                                'prop_value': label_value
                                            }
                                    else:
                                        if value == label_value[:4]:
                                            if is_ceaTask: return f"http://www.wikidata.org/entity/{entity_id}"
                                            return {
                                                'prop_id': property_id,
                                                'prop_value': label_value
                                            }
                                        _values = []
                                        for item in property_values:
                                            if ((len(item.split("-")) or len((item.split("/"))) == 3 and (item.split['-'][0].isdigit()) or item.split['/'][0].isdigit())):
                                                _values.append(item[0:4])
                                            else:
                                                _values.append(item)
                                        if value in _values:
                                            property_values = _values
                                elif value[0:4].isdigit():
                                    if is_horizontal and not is_combined:
                                        if value[:4] == label_value[:4]:
                                            print(f"http://www.wikidata.org/entity/{entity_id}")
                                            return {
                                                'prop_id': property_id,
                                                'prop_value': label_value
                                            }
                                    else:
                                        if value[:4] in label_value:
                                            if is_ceaTask: return f"http://www.wikidata.org/entity/{entity_id}"
                                            return {
                                                'prop_id': property_id,
                                                'prop_value': label_value
                                            }
                                        _values = value[0:7]
                                        for item in property_values:
                                            if _values in item or value[0:4] in item:
                                                index = property_values.index(item)
                                                property_values[index] = value
                                                break
                        property_value_ids.append(claim["mainsnak"]["property"])

    return None   

def check_entity_properties_cea(entity_ids, property_values, is_column_id=True, label_current=None, context_have_string_value=False):
    if len(property_values) == 0:
        return entity_ids
    property_values = property_values[1:]
    if (len(entity_ids) <= 3 and len(entity_ids) != 0):
        return entity_ids[0]
    print(len(entity_ids))

    for entity_id in entity_ids:
        try:
            url = f"https://www.wikidata.org/w/api.php?action=wbgetclaims&entity={entity_id}&format=json"
            response = requests.get(url)
            data = response.json()
            if "claims" in data:
                claims = data["claims"]
                entity_property_values = []
                property_ids = []
                property_values_label = []

                for property_id, property_claims in claims.items():
                    i = 1
                    for claim in property_claims:
                        if "mainsnak" in claim and "datavalue" in claim["mainsnak"]:
                            value = claim["mainsnak"]["datavalue"]["value"]
                            if 'id' in value and not isinstance(value, str):
                                    # print(value['id'])
                                    result = get_entity_with_id(value['id'])
                                    label = result['label'] if 'label' in result else ''
                                    if label in property_values:
                                        entity_property_values.append(label)
                                    else:
                                        continue
                                    if i == len(property_values):
                                        break
                                    else:
                                        i += 1
                                        continue
                            
                            elif 'amount' in value and not isinstance(value, str):
                                value = value['amount']
                                entity_property_values.append(value)
                            elif 'longitude' and 'latitude' in value and not isinstance(value, str):
                                precision = value['precision']
                                # print(value)
                                if precision and  precision > 0:
                                    longitude = round(float(value['longitude']), int(-math.log10(precision)))
                                    latitude = round((value['latitude']), int(-math.log10(precision)))
                                else:
                                    longitude = value['longitude']
                                    latitude = value['latitude']
                                value = f"Point({longitude}, {latitude})"
                                entity_property_values.append(value)
                            elif 'time' in value and not isinstance(value, str):
                                value = value['time'][1:11]
                                entity_property_values.append(value)
                            elif 'text' in value and not isinstance(value, str):
                                value = value['text']
                                entity_property_values.append(value)
                            else:
                                entity_property_values.append(value)
                            if property_id == "P571":
                                if value in property_values:
                                    pass
                                elif value.isdigit():
                                    _values = []
                                    for item in property_values:
                                        if ((len(item.split("-")) or len((item.split("/"))) == 3 and (item.split['-'][0].isdigit()) or item.split['/'][0].isdigit())):
                                            _values.append(item[0:4])
                                        else:
                                            _values.append(item)
                                    if value in _values:
                                        property_values = _values
                                elif value[0:4].isdigit():
                                    _values = value[0:7]
                                    for item in property_values:
                                        if _values or value[0:4] in item:
                                            index = property_values.index(item)
                                            property_values[index] = value
                                            break
                # print(entity_property_values)
                property_values = is_subpart(property_values, entity_property_values) 
                converted_data = [frozenset(item.items()) if isinstance(item, dict) else item for item in entity_property_values]
                if set(property_values).issubset(set(converted_data)):
                    print("Yes")
                    # print(entity_property_values)
                    return entity_id

        except:
            print("Not entity")
    if len(entity_ids) != 0 and isinstance(entity_ids, list):
        return entity_ids[0]
    return None


# get wikidata entity with his Qid
def get_entity_properties(entity_id):
    base_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "format": "json"
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    # Check response and extract properties
    if "entities" in data and entity_id in data["entities"]:
        entity = data["entities"][entity_id]
        if "claims" in entity:
            properties = entity["claims"]['P17']
            return properties

    return []


# this function allow us to check if a wikidata property contain the
# values passed in input
def check_property_contains_value(entity_id, value):
    base_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "format": "json",
        "props": "claims"
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    # Check the answer and search property conteninf the input value
    if "entities" in data and entity_id in data["entities"]:
        entity = data["entities"][entity_id]
        if "claims" in entity:
            for property_id, claims in entity["claims"].items():
                for claim in claims:
                    if "mainsnak" in claim and "datavalue" in claim["mainsnak"]:
                        datavalue = claim["mainsnak"]["datavalue"]
                        if "value" in datavalue and value.lower() in str(datavalue["value"]).lower():
                            return property_id

    return None



def get_property_from_value(value):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    # Construction de la requête SPARQL
    query_template = """
    SELECT DISTINCT ?entity ?property ?propertyLabel WHERE {{
      ?entity ?property ?value.
      VALUES ?value {{ "{0}" }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    """
    try:

        query = query_template.format(value)

        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)

        # run sparql query
        results = sparql.query().convert()

        # get property
        property_label = None
        list_entity = []
        # print(results["results"]["bindings"])
        if len(results["results"]["bindings"]) > 0:
            results = results["results"]["bindings"]
            if len(results) > 1:
                for props in results:
                    if "direct" in props["property"]["value"].split("/"):
                        # property_id = props["property"]["value"].split("/")[-1]
                        property_label = props["propertyLabel"]["value"]
                        entity_label = props['entity']['value'].split("/")[-1].split("-")[0]
                        if entity_label.lower().startswith("q"):
                            list_entity.append((entity_label, property_label))
            else:
                for props in results:
                    # property_id = props["property"]["value"].split("/")[-1]
                    property_label = props["propertyLabel"]["value"]
                    entity_label = props['entity']['value'].split("/")[-1].split("-")[0]
                    if entity_label.lower().startswith("q"):
                        list_entity.append((entity_label, property_label))
            # list_entity = [entity[0] for entity in list_entity]
            print(f"len list-entity = {len(list_entity)}")
            if len(list_entity) > 5:
                list_entity = None
        else:
            list_entity = None
        return list_entity
    except:
        return None


def is_subset(sublist, mainlist):
    found_subset = False
    for item in sublist:
        if item in mainlist:
            found_subset = True
            print(item)
            continue
        else:
            for subitem in mainlist:
                if isinstance(subitem, list):
                    if item in subitem:
                        found_subset = True
                        print("Very True")
                        break
                    elif item == ",".join(subitem):
                        found_subset = True
                        break
                    else:
                        found_subset = False
                else:
                    found_subset = False
    return found_subset

def get_properties(values: list[str], is_horizontal=False, label=None, index=None):
    new_val = []
    is_combined = False
    properties = ''
    for val in values:
        if val.startswith(" "):
            val = val[1:]
        if val.endswith(" "):
            val = val[:-1]
        new_val.append(val)
    values = new_val
    print(values)
    for value in values:
        if is_number(str(value)):
            if str(value).split(".")[-1] == 0:
                value = str(value).split(".")[0]
                value = f"+{value}"
            else:
                value = f"+{value}"
        else:       
            value = value.strip()
        if "//" in value:
            value = value.split('/')[-1]
            if ":" in value:
                value = value.split(":")[-1]
        entity_ids = get_property_from_value(value)
        if (entity_ids and len(entity_ids) == 1 and value == label):
            return {
                'prop_id': entity_ids[0][1],
                'prop_value': label
            }
        elif entity_ids:
            entity_ids = [idx[0] for idx in entity_ids]
        if not entity_ids:
            continue
        else:
            print(f"value = {value}, -> entity {entity_ids}, label_to_check={label}")
            if len(entity_ids) > 1:
                is_combined = True        
            properties = check_entity_properties_cpa(entity_ids, values, is_horizontal, label, index=index)
            if not properties:
                continue
            else:
                print(properties)
                break
    return properties

# def get_entities_with_matching_properties(entity_ids, values, is_horizontal=False, label_value=None):
#     base_url = "https://www.wikidata.org/w/api.php"
#     params = {
#         "action": "wbgetentities",
#         "ids": "|".join(entity_ids),
#         "format": "json",
#         "props": "claims"
#     }
#     if is_horizontal:
#         context_have_string_value = have_string(label_value)
#     else:
#         context_have_string_value = have_string(values)
#     try:
#         response = requests.get(base_url, params=params)
#         data = response.json()

#         # check and looking for entities with matching properties
#         if "entities" in data:
#             for entity_id, entity_data in data["entities"].items():
#                 print(entity_id)
#                 if "claims" in entity_data:
#                     entity_properties = [] # contains properties_ids and properties labels
#                     property_values_label = [] # properties labels
#                     property_values_id = [] # properties ids
#                     for property_id, claims in entity_data["claims"].items():
#                         for claim in claims:
#                             # print(claim)
#                             if "mainsnak" in claim and "datavalue" in claim["mainsnak"]:
#                                 datavalue = claim["mainsnak"]["datavalue"]
#                                 if "value" in datavalue:
#                                     property_value = datavalue["value"]
#                                     if is_horizontal:
#                                         if label == property_value:
#                                             return {
#                                                 'prop_id': property_id,
#                                                 'prop_value': label_value
#                                             }
#                                     # check if property_value is json object rather than string object
#                                     if 'id' in property_value and not isinstance(property_value, str):
#                                         # use id to get the label of the entity that match with this id
#                                         result = get_entity_with_id(property_value['id'].split("$")[0])
#                                         label = result['label'] if 'label' in result else ''
#                                         ids = result['id']
#                                         property_value = (label)
#                                     property_values_id.append(property_id)
#                                     property_values_label.append(property_value)
#                                     new_label = []
#                                     # Join all value in label if it contain(,)
#                                     for label in property_values_label:
#                                         try:
#                                             if float(label) or int(label):
#                                                 label = f"+{label}"
#                                         except:
#                                             label = label
#                                         if ',' in label:
#                                             new_label.append(label.split(","))
#                                         else:
#                                             new_label.append(label)
#                                     # new list of the values of entity
#                                     property_values_label = new_label
#                     entity_properties.append(property_values_label)
#                     entity_properties.append(property_values_id)
#                     print(entity_properties[0])
#                     converted_data = [frozenset(item.items()) if isinstance(item, dict) else item for item in entity_properties[0]]
#                     # print(converted_data)
#                     try:
#                         if is_subset(values, converted_data):
#                             property_ids = []
#                             indeces = [entity_properties[0].index(item) for item in values]
#                             print(indeces)
#                             for i in indeces:
#                                 property_ids.append(entity_properties[1][i])
                            
#                             entity_properties = []
#                             i = 0
#                             for elt in property_ids:
#                                 entity_properties.append({
#                                     'prop_id': elt,
#                                     'prop_value': values[i]
#                                 })
#                                 i += 1
#                             print(entity_properties)
#                             break
#                         else:
#                             print("None")
#                             entity_properties = None
#                     except TypeError as e:
#                         entity_properties = None
#                         print("error",e)

#         return entity_properties
#     except ValueError as e:
#         print(e)

