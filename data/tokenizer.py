from dataclasses import dataclass
from urllib.parse import urlparse
from typing import Any, List, Tuple
import re
from random import choice
from pickle import dump



tokens = {
    'start_label': re.compile('<[^>]+>'),
    'end_label': re.compile('</[^>^/]+>'),
    'events': re.compile('on[a-zA-Z]+='),
    'function_name': re.compile('(?:[a-zA-Z$_][a-zA-Z$_0-9]*\.)*[a-zA-Z$_][a-zA-Z$_0-9]*\('),
    'identifier': re.compile('(?:[a-zA-Z$_][a-zA-Z$_0-9]*\.)*[a-zA-Z$_][a-zA-Z$_0-9]*'),
    'string_arg' : re.compile('\s*(\"(?:(\\(\\|n|r|\')|[^\\\n\r\"])*\"|\'(?:(\\(\\|n|r|\")|[^\\\n\r\'])*\')\s*\)'),
    'num_arg' : re.compile('[1-9][0-9]*\.?[0-9]*\)'),
    'assign_constant': re.compile('=(0|[1-9][0-9]*\.?[0-9]*)|=(\"(?:(\\(\\|n|r|\')|[^\\\n\r\"])*\"|\'(?:(\\(\\|n|r|\")|[^\\\n\r\'])*\')'),
    'script_url': re.compile('[a-zA-Z]*script:'),
    'other': re.compile('\{\[|\]|\.|\.\.\.|\;|\,|<|>|<=|>=|==|!=|===|!==|\+|-|\*|%|\*\*|\+\+|--|<<|>>|>>>|&|\||\^|!|~|&&|\|\||\?\?|\?|:|=|\+=|-=|\*=|%=|\*\*=|<<=|>>=|>>>=|&=|\|=|\^=|&&=|\|\|=|\?\?=|=>|/')
}

@dataclass
class JSToken:
    type: str
    value: str
    start: int
    end: int

@dataclass
class URLTokens:
    parsed_url: str
    token_list: List[JSToken]



def trim_url(url: str):
    url = urlparse(url)
    return url.path + '?' + url.query + '#' + url.fragment

def ordered_interval_overlaps(first: Tuple[int, int], second: Tuple[int, int] ) -> bool:
    a, b = first
    c, _ = second

    # if the start of the second is inside the first
    return c >= a and c < b

def prune_tokens(sorted_tokens: List[JSToken]) -> List[JSToken]:
    sorted_tokens.sort(key = lambda x: (x.start, -x.end))
    curr = sorted_tokens[0]
    pruned_tokens = list([curr])

    for token in sorted_tokens[1:]:
        if ordered_interval_overlaps( (curr.start, curr.end), (token.start, token.end) ):
            pass
        else:
            pruned_tokens.append(token)
            curr = token
    return pruned_tokens 

def tokenize(url: str) -> List[JSToken]:

    url = trim_url(url)       
    token_list = list()
    for token_type in tokens:
        for matched in tokens[token_type].finditer(url):
            token_list.append( ( JSToken(token_type, matched.group(0), matched.start(), matched.end()) ) )

    sorted_tokens = prune_tokens(token_list)
    return sorted_tokens


def print_token_list(url_tok: URLTokens):
    print(url_tok.parsed_url)
    for token in url_tok.token_list:
        print(token)
    print('')

if __name__ == "__main__":

    # with open('dec_xss_urls.txt', 'r') as infile:
    #     lines = infile.read().splitlines()
    #     for i in range(10):
    #         url = choice(lines)
    #         tmp = URLTokens(url, tokenize(url))
    #         print_token_list(tmp)
            # dump(tmp, outfile)

    url = 'http://website/remind.php?l=fr&error=2&email=/"><script>alert(\'Xss ByAtm0n3r\')</script><script type="text/javascript"src="http://yourjavascript.com/27544112151/xss.atmon3r.js"></script>'
    tmp = URLTokens(url, tokenize(url))
    print_token_list(tmp)
        # for line in lines:
        #     tmp = JSTokenizer(DeepURL(line))

    # tmp = JSTokenizer(DeepURL('http://website/projects/eva/details.php?ta="><img src=vbscript:document.write("By_SecreT")><script>alert(2)</script>'))
