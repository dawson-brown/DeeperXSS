from dataclasses import dataclass
from urllib.parse import urlparse
from typing import Any, List, Tuple
import re
from random import choice
from pickle import dump
from sys import argv


# 'num_arg' : re.compile('[0-9]+\.?[0-9]*\)'),
# 'function_name': re.compile('(?:[a-zA-Z$_][a-zA-Z$_0-9]*\.)*[a-zA-Z$_][a-zA-Z$_0-9]*\('),
tokens = {
    'start_label': re.compile('<[^>\s/]+(>|\s)'),
    'end_label': re.compile('</[^>/]+>'),
    'events': re.compile('on[a-zA-Z]+='),
    'identifier': re.compile('(?:[a-zA-Z$_][a-zA-Z$_0-9]*\.)*[a-zA-Z$_][a-zA-Z$_0-9]*'),
    # 'string_arg' : re.compile('\s*(\"(?:(\\(\\|n|r|\')|[^\\\n\r\"])*\"|\'(?:(\\(\\|n|r|\")|[^\\\n\r\'])*\')\s*\)'),
    'int_constant': re.compile('[0-9]+\.?[0-9]*'),
    'other': re.compile('\{\[|\]|\.|\;|\,|<|>|=|!|\+|-|\*|%|&|\||\^|~|\?|:|/|\'|"')
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
                token_list.append( ( JSToken(token_type, matched.group(0)[:-1] + '>', matched.start(), matched.end()) ) )
            elif token_type == 'identifier':
                if matched.end() < len(url) and url[matched.end()]  == '(':
                    token_list.append( ( JSToken('function_name', matched.group(0), matched.start(), matched.end()) ) )
                    arg_end = url.find(')', matched.end())
                    if arg_end != -1:
                        token_list.append( JSToken('args', url[matched.end():arg_end+1], matched.end(), arg_end+1) )
                else:
                    token_list.append( ( JSToken(token_type, matched.group(0), matched.start(), matched.end()) ) )
            else:
                token_list.append( ( JSToken(token_type, matched.group(0), matched.start(), matched.end()) ) )

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
            # print_token_list(tmp)
            dump(tmp, outfile)

if __name__ == "__main__":

    # print('XSSed...')
    # with open('dec_xss_urls.txt', 'r') as infile, \
    #     open('dec_xss_urls.dat', 'ab') as outfile:
    #     lines = infile.read().splitlines()
    #     for i in range(10):
    #         url = choice(lines)
    #         tmp = URLTokens(url, tokenize(url))
    #         print_token_list(tmp)
            # dump(tmp, outfile)

    start_i = 0
    end_i = -1
    if len(argv) > 1:
        start_i = argv[1]

        if len(argv) > 2:
            end_i = argv[2]

    print(f'XSSed...{start_i} - {end_i}')
    tokenize_to_file('dmoz_dir.txt', int(start_i), int(end_i))

    # url = 'http://website/Hotel-Search?action=hotelSearchWizard@searchHotelOnly&hotelSearchWizard_inpItid=&hotelSearchWizard_inpItty=&hotelSearchWizard_inpItdx=&hotelSearchWizard_inpSearchMethod=usertyped&hotelSearchWizard_inpSearchKeywordIndex=&hotelSearchWizard_inpSearchKeyword=&hotelSearchWizard_inpSearchRegionId=&hotelSearchWizard_inpSearchLatitude=&hotelSearchWizard_inpSearchLongitude=&hotelSearchWizard_inpSearchNear=/"><script>alert(\'Xss ByAtm0n3r\')</script>&hotelSearchWizard_inpSearchNearType=CITY&hotelSearchWizard_inpSearchNearStreetAddr=&hotelSearchWizard_inpSearchNearCity=&hotelSearchWizard_inpSearchNearState=&hotelSearchWizard_inpSearchNearZipCode=&hotelSearchWizard_inpCheckIn=jj/mm/aa&hotelSearchWizard_inpCheckOut=jj/mm/aa&hotelSearchWizard_inpNumRooms=1&hotelSearchWizard_inpNumAdultsInRoom=1&hotelSearchWizard_inpNumChildrenInRoom=0&hotelSearchWizard_inpAddOptionFlag=&hotelSearchWizard_inpHotelName=&hotelSearchWizard_inpHotelClass=0&searchWizard_wizardType=hotelOnly'
    # tmp = URLTokens(url, tokenize(url))
    # print_token_list(tmp)
        