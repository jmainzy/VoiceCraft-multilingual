import re

chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\[\]\(\)\$\-_»:«]'
a_nasal='ą'
e_nasal='ę'
u_nasal='ų'
o_nasal='ǫ'

def remove_special_chars(batch):    
    batch['text'] = re.sub(chars_to_remove_regex, '', batch['text']).lower()
    
    return batch