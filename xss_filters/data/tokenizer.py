from dataclasses import dataclass
from urllib.parse import urlparse
from typing import Any, List, Tuple
import re
from random import choice
from pickle import dump
from sys import argv



tokens = {
    'start_label': re.compile('<[^>\s/]+(>|\s)'),
    'end_label': re.compile('</[^>/]+>'),
    'events': re.compile('on[a-zA-Z]+='),
    'function_name': re.compile('(?:[a-zA-Z$_][a-zA-Z$_0-9]*\.)*[a-zA-Z$_][a-zA-Z$_0-9]*\('),
    'identifier': re.compile('(?:[a-zA-Z$_][a-zA-Z$_0-9]*\.)*[a-zA-Z$_][a-zA-Z$_0-9]*'),
    'string_arg' : re.compile('\s*(\"(?:(\\(\\|n|r|\')|[^\\\n\r\"])*\"|\'(?:(\\(\\|n|r|\")|[^\\\n\r\'])*\')\s*\)'),
    'num_arg' : re.compile('(0|[1-9][0-9]*)\.?[0-9]*\)'),
    # 'assign_constant': re.compile('=(0|[1-9][0-9]*\.?[0-9]*)|=(\"(?:(\\(\\|n|r|\')|[^\\\n\r\"])*\"|\'(?:(\\(\\|n|r|\")|[^\\\n\r\'])*\')'),
    'int_constant': re.compile('(0|[1-9][0-9]*)'),
    'script_url': re.compile('[a-zA-Z]*script:'),
    'other': re.compile('\{\[|\]|\.|\.\.\.|\;|\,|<|>|<=|>=|==|!=|===|!==|\+|-|\*|%|\*\*|\+\+|--|<<|>>|>>>|&|\||\^|!|~|&&|\|\||\?\?|\?|:|=|\+=|-=|\*=|%=|\*\*=|<<=|>>=|>>>=|&=|\|=|\^=|&&=|\|\|=|\?\?=|=>|/|\'|"')
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
    trimmed_url = url.path
    if url.query != '':
        trimmed_url = trimmed_url + '?' + url.query + '#' + url.fragment
    return trimmed_url

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
            if token_type == 'start_label':
                match_str = matched.group(0)[:-1] + '>'
            else: 
                match_str = matched.group(0)
            token_list.append( ( JSToken(token_type, match_str, matched.start(), matched.end()) ) )

    sorted_tokens = prune_tokens(token_list)
    return sorted_tokens


def print_token_list(url_tok: URLTokens):
    print(url_tok.parsed_url)
    for token in url_tok.token_list:
        print(token)
    print('')

def tokenize_to_file(in_name: str, start_line = 0, end_line = -1):

    out_name = f'{in_name}_{start_line}-{end_line}.dat'

    with open(in_name, 'r') as infile, \
        open(out_name, 'ab') as outfile:
        lines = infile.read().splitlines()

        if end_line == -1:
            end_line = len(lines)
        for i, url in enumerate(lines[start_line:end_line]):
            if i%100 == 0:
                print(f'\tLine: {i}')
            tmp = URLTokens(url, tokenize(url))
            dump(tmp, outfile)

if __name__ == "__main__":

    # print('DMOZ...')
    # with open('dmoz_dir.txt', 'r') as infile, \
    #     open('dmoz_tokens.dat', 'ab') as outfile:
    #     lines = infile.read().splitlines()
    #     for i, url in enumerate(lines):
    #         if i%100 == 0:
    #             print(f'\tLine: {i}')
    #         tmp = URLTokens(url, tokenize(url))
    #         dump(tmp, outfile)

    start_i = 0
    end_i = -1
    if len(argv) > 1:
        start_i = argv[1]

        if len(argv) > 2:
            end_i = argv[2]

    print(f'XSSed...{start_i} - {end_i}')
    tokenize_to_file('dec_xss_urls.txt', int(start_i), int(end_i))

    # url = 'http://website/5700-4-0-1.html?path=http://www.zdnet.com/1383-4-44.html?path=http://talkback.zdnet.com/5208-12558-0-1.html?siteID=24&forumID=1&threadID=53856&ct=null&messageID=1019276&start=-1&reply=true&subject=RE: Microsoft ranks Windows 7 features most likely to affect app-compatibility®ister=true&email="><script src = \'http://www.reelix.za.net/reexss.js\'></script>&password=&passwordConf=&username=&firstName=&lastName=&company=&address1=&address2=&city=&country=US&state=&postalCode=&phone=&JOBCAT=NOTSELECTED&jobFunction=NOTSELECTED&industry=NOTSELECTED&companySize=NOTSELECTED&newsletters=e580:INTERNAL_NEWSLETTER&newsletters=e590:INTERNAL_NEWSLETTER&newsletters=e589:INTERNAL_NEWSLETTER&newsletters=e539:INTERNAL_NEWSLETTER&newsletters=e550:INTERNAL_NEWSLETTER&rememberMe=true&submit.x=63&submit.y=10'
    # tmp = URLTokens(url, tokenize(url))
    # print_token_list(tmp)
        # for line in lines:
        #     tmp = JSTokenizer(DeepURL(line))

    # tmp = JSTokenizer(DeepURL('http://website/projects/eva/details.php?ta="><img src=vbscript:document.write("By_SecreT")><script>alert(2)</script>'))
