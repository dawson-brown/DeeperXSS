from dataclasses import dataclass
from xssed_url_clean import DeepURL
from urllib.parse import ParseResult, urlparse
from typing import Any, List, Tuple
import re
from random import choice
from pickle import dump, load



tokens = {
    'start_label': re.compile('<[^>/\s]+>?'),
    'end_label': re.compile('</[^>^/]+>'),
    'events': re.compile('on[a-zA-Z]*='),
    'function_name': re.compile('(?:[a-zA-Z$_][a-zA-Z$_0-9]+\.)*[a-zA-Z$_][a-zA-Z$_0-9]+\('),
    'variable': re.compile('(?:[a-zA-Z$_][a-zA-Z$_0-9]+\.)*[a-zA-Z$_][a-zA-Z$_0-9]+\(?'),
    'string_arg' : re.compile('\(\w*\"(?:(\\(\\|n|r|\')|[^\\\n\r\'])*\"\w*\)'),
    'num_arg' : re.compile('\([1-9][0-9]*.?[0-9]*\)'),
    'script_url': re.compile('[a-zA-Z]*script:'),
    'other': re.compile('\{\[|\]|\.|\.\.\.|\;|\,|<|>|<=|>=|==|!=|===|!==|\+|-|\*|%|\*\*|\+\+|--|<<|>>|>>>|&|\||\^|!|~|&&|\|\||\?\?|\?|:|=|\+=|-=|\*=|%=|\*\*=|<<=|>>=|>>>=|&=|\|=|\^=|&&=|\|\|=|\?\?=|=>')
}

@dataclass
class JSToken:
    value: str
    start: int
    end: int

@dataclass
class URLTokens:
    parsed_url: str
    token_list: List[Tuple[str, JSToken]]



def trim_url(url: str):
    url = urlparse(url)
    return url.path + '?' + url.query + '#' + url.fragment

def ordered_interval_contains(first: Tuple[int, int], second: Tuple[int, int] ) -> bool:
    a, b = first
    c, d = second

    return c >= a and d <= b

def prune_tokens(sorted_tokens: List[Tuple[str, JSToken]]) -> List[Tuple[str, JSToken]]:
    sorted_tokens.sort(key = lambda x: (x[1].start, -x[1].end))
    curr = sorted_tokens[0]
    pruned_tokens = list([curr])

    for token in sorted_tokens[1:]:
        if ordered_interval_contains( (curr[1].start, curr[1].end), (token[1].start, token[1].end) ):
            pass
        else:
            pruned_tokens.append(token)
            curr = token
    return pruned_tokens 

def tokenize(url: str) -> List[Tuple[str, JSToken]]:

    url = trim_url(url)       
    token_list = list()
    for token_type in tokens:
        for matched in tokens[token_type].finditer(url):
            token_list.append( ( token_type, JSToken(matched.group(0), matched.start(), matched.end()) ) )

    sorted_tokens = prune_tokens(token_list)
    return sorted_tokens


if __name__ == "__main__":

    with open('dec_xss_urls.txt', 'r') as infile, \
            open('xss_tokens.txt', 'ab') as outfile:
        lines = infile.read().splitlines()
        for i in range(10):
            url = choice(lines)
            tmp = URLTokens(url, tokenize(url))
            dump(tmp, outfile)

    # print('\n\nPickle Load:\n')
    # with open('xss_tokens.txt', "rb") as f:
    #     while True:
    #         try:
    #             tmp = load(f)
    #         except EOFError:
    #             break

        # for line in lines:
        #     tmp = JSTokenizer(DeepURL(line))

    # tmp = JSTokenizer(DeepURL('http://website/projects/eva/details.php?ta="><img src=vbscript:document.write("By_SecreT")><script>alert(2)</script>'))
