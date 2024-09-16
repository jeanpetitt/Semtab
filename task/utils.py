import re


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

def find_element_or_first(lst):
    for element in lst:
        if re.match(r'^Q\d+$', element):  # Regex to match 'Q' followed by one or more digits
            return element
    return lst[0] if lst else None

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

def is_valid_string(s):
    # This pattern checks that the string does not contain ?, @, #, or $ in invalid positions
    pattern = r"^[^?@#$][^\?@#$]*[^?@#$]?$"
    return bool(re.match(pattern, s))