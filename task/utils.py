import re
from urllib.parse import unquote
import random

random.seed(42)

def is_number(string):
    word = "".join(str(string).split())[0:6]
    if word.isdigit():
        return True
    return re.match(r"^[+-]?\d+(\.\d+)?$", str(string)) is not None


def is_date(string):
    split_slash = str(string).split("/")
    split_tiret = str(string).split("-")
    if len(split_slash) == 3 or len(split_tiret) == 3:
        if is_number(split_tiret[0]) or is_number(split_tiret[0][1:]) or is_number(split_slash[0]):
            return True
    return False  

def find_element_or_choice(lst):
    for element in lst:
        if re.match(r'^q\d+$', element.lower()):  # Regex to match 'Q' followed by one or more digits
            return element
    return random.choice(lst) if lst else None

def contains_html_tags(text):
    # Define a regex pattern for HTML tags
    html_tag_pattern = re.compile(r'<[^>]+>')
    # Search for the pattern in the input text
    if html_tag_pattern.search(text):
        return True
    return False

def is_valid_format(s):
    # This pattern matches a word followed by square brackets containing any content
    pattern = r"^\w+\[.*\]$"
    return bool(re.match(pattern, s))

def correct_string(expression: str) -> str:
    stack = []
    result = []
    inside_quotes = False
    escaped = False

    # Parcourir chaque caractère de la chaîne
    for char in expression:
        # Gérer les caractères échappés
        if char == "\\" and not escaped:
            result.append(char)
            escaped = True
            continue

        # Vérifier si on est à l'intérieur des guillemets
        if char == '"' and not escaped:
            inside_quotes = not inside_quotes  # Alterner l'état à chaque guillemet
            result.append(char)
            continue
        
        escaped = False  # Réinitialiser l'échappement après chaque caractère

        # Gérer les crochets et les parenthèses si on n'est pas dans une chaîne de caractères
        if not inside_quotes:
            if char == '[':
                stack.append(char)
                result.append(char)
            elif char == ']':
                if stack and stack[-1] == '[':
                    stack.pop()  # Correspondance correcte
                else:
                    result.append('[')  # Ajouter un crochet ouvrant s'il manque
                result.append(char)
            else:
                result.append(char)
        else:
            result.append(char)

    # Ajouter les crochets ou parenthèses manquants après la fin de la chaîne
    while stack:
        opening = stack.pop()
        if opening == '[':
            result.append(']')
    # print(result)
    corrected_string = ''.join(result)

    # Corriger les guillemets doubles à l'intérieur des crochets pour qu'ils deviennent des guillemets simples
    corrected_string = re.sub(r'\["(.*?)"\]', r"['\1']", corrected_string)

    # Séparer les mots collés avec des majuscules
    corrected_string = re.sub(r'(?<!^)(?=[A-Z])', ' ', corrected_string)

    return corrected_string

def is_valid_string(s):
    # This pattern checks that the string does not contain , @, #, or $ in invalid positions
    pattern = r"^[^?@#$][^\?@#$]*[^?@#$]?$"
    return bool(re.match(pattern, s))


def decole_url_file(url_file: str):
    filename = url_file.split("/")[-1]
    decoded_filename = unquote(filename)
    return decoded_filename

def is_numerical(object: list):
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

def choose_random_valid_element(object: list):
    valid_elements = [element for element in object if element not in (None, "", False)]
    if not valid_elements:
        return None

    return random.choice(valid_elements)
# print(decole_url_file("http://commons.wikimedia.org/wiki/Special:FilePath/JAS%2039%20Gripen%20momument%20L%C3%A5ngholmen%202.jpg"))