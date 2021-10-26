from enum import Enum
from base64 import b64decode
from urllib import parse
from html import unescape

class EType(Enum):
    USASCII = 0
    UTF7 = 1
    URL = 2
    HTML = 3
    DWORD = 4
    HEX = 5
    OCTAL = 6
    BASE64 = 7


def de_encoding_type(string : str) -> EType:
    pass
    
def de_us_ascii_decode(string : str) -> str:
    pass

def de_utf7_decode(string : str) -> str:
    pass

def de_url_decode(string : str) -> str:
    if string.find('%') == -1:
        raise ValueError
    return parse.unquote(string)

def de_html_decode(string : str) -> str:
    return unescape(string)

def de_dword_decode(string : str) -> str:
    pass

def de_hex_decode(string : str) -> str:
    return bytes.fromhex(string).decode("utf-8")

def de_octal_decode(string : str) -> str:
    pass

def de_base64_decode(string : str) -> str:
    string = string.replace('-', '=')\
            .replace('.', '+')\
            .replace('_', '/')

    return b64decode(string).decode("utf-8")

def de_dfs_decode(string : str) -> str:
    pass
