from dataclasses import dataclass
from urllib.parse import urlparse
from typing import Any, List, Tuple
import re
from random import choice
from random import randrange
from pickle import dump, load
from sys import argv
import time



# 'num_arg' : re.compile('[0-9]+\.?[0-9]*\)'),
# 'function_name': re.compile('(?:[a-zA-Z$_][a-zA-Z$_0-9]*\.)*[a-zA-Z$_][a-zA-Z$_0-9]*\('),
tokens = {
    'start_label': re.compile('<[^>\s/]+(>|\s|/)'),
    'end_label': re.compile('</[^>/]+>'),
    'events': re.compile('on[a-zA-Z]+='),
    'identifier': re.compile('(?:[a-zA-Z$_0-9\-]+\.)*[a-zA-Z$_][a-zA-Z$_0-9\-]*'),
    # 'string_arg' : re.compile('\s*(\"(?:(\\(\\|n|r|\')|[^\\\n\r\"])*\"|\'(?:(\\(\\|n|r|\")|[^\\\n\r\'])*\')\s*\)'),
    'int_constant': re.compile('[0-9]+\.?[0-9]*'),
    'other': re.compile('<|>|\+|-|\*')
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
    if (len(sorted_tokens)) == 0:
        return None
    curr = sorted_tokens[0]
    pruned_tokens = list([curr])

    for token in sorted_tokens[1:]:
        if ordered_interval_overlaps( (curr.start, curr.end), (token.start, token.end) ):
            pass
        else:
            pruned_tokens.append(token)
            curr = token
    return pruned_tokens 

def tokenize(url_full: str) -> List[JSToken]:

    url = trim_url(url_full)       
    token_list = list()

    num_paths = 0
    num_assign = 0
    num_vals = 0
    num_idens = 0

    for token_type in tokens:
        for matched in tokens[token_type].finditer(url):
            if token_type == 'start_label':
                tokens_to_append = [JSToken(token_type, (matched.group(0)[:-1] + '>').lower(), matched.start(), matched.end()) ]
            elif token_type == 'identifier':
                if matched.end() < len(url) and url[matched.end()]  == '(':
                    tokens_to_append = [JSToken('function_name', matched.group(0).lower(), matched.start(), matched.end()) ]
                    arg_end = url.find(')', matched.end())
                    if arg_end != -1:
                        args = url[matched.end()+1:arg_end].strip()
                        if args.isdigit():
                            tokens_to_append.append(JSToken('int_arg', '(1)', matched.end(), arg_end+1))
                        else:
                            tokens_to_append.append(JSToken('str_arg', '("str_arg")', matched.end(), arg_end+1))
                elif matched.end() < len(url) and url[matched.end()]  == '=':
                    tokens_to_append = [JSToken('assignment', f'assign{num_assign}' + '=', matched.start(), matched.end())]
                    num_assign+=1
                elif url[matched.start()-1]  == '=':
                    tokens_to_append = [JSToken('value', f'val{num_vals}', matched.start(), matched.end())]
                    num_vals+=1
                elif matched.end() < len(url) and url[matched.end()]  == ':' and matched.group(0).lower().endswith('script'):
                    tokens_to_append = [JSToken('script_url', matched.group(0).lower(), matched.start(), matched.end())]
                elif matched.end() < len(url) and (url[matched.end()]  == '/' or url[matched.end()]  == '?'):
                    tokens_to_append = [JSToken('path', f'path{num_paths}' + '/', matched.start(), matched.end())]
                    num_paths+=1
                elif matched.end() == len(url) and url[matched.start() - 1] == '/':
                    tokens_to_append = [JSToken('path', matched.group(0).lower(), matched.start(), matched.end())]
                else:
                    tokens_to_append = [JSToken('identifier', f'identifier{num_idens}', matched.start(), matched.end())]
                    num_idens+=1
            elif token_type == 'int_constant':
                tokens_to_append = [JSToken(token_type, '1', matched.start(), matched.end())]
            else:
                tokens_to_append = [JSToken(token_type, matched.group(0).lower(), matched.start(), matched.end())]

            for token in tokens_to_append:
                token_list.append(token)

    sorted_tokens = prune_tokens(token_list)
    return sorted_tokens


def load_tokenized_urls(filename: str) -> URLTokens:

    with open(filename, "rb") as f:
        while True:
            try:
                yield load(f)
            except EOFError:
                break


def longest_url(filename: str) -> int:
    longest = 0
    for tokenized_url in load_tokenized_urls(filename):
        if len(tokenized_url.token_list) > longest:
            longest = len(tokenized_url.token_list)
    return longest  

def print_token_list(url_tok: URLTokens):
    print(url_tok.parsed_url)
    for token in url_tok.token_list:
        print(token)
    print('')

def tokenize_to_std(in_name: str, num = 0):

    with open(in_name, 'r') as infile:
        lines = infile.read().splitlines()

        start = randrange(len(lines))
        step = randrange(100)
        print(f'{start} - {step}')

        for i in range(num):
            url = lines[start]
            start = (start + step) % len(lines)
            tmp = URLTokens(url, tokenize(url))
            print_token_list(tmp)

def tokenize_to_file(in_name: str, start_line = 0, end_line = -1):

    timestr = time.strftime("%Y%m%d-%H%M%S")
    out_name = f'{in_name}__{timestr}_{start_line}-{end_line}.dat'

    with open(in_name, 'r') as infile, \
        open(out_name, 'ab') as outfile:
        lines = infile.read().splitlines()

        if end_line == -1:
            end_line = len(lines)
        for i, url in enumerate(lines[start_line:end_line]):
            if i%100 == 0:
                print(f'\tLine: {i}')
            tmp = URLTokens(url, tokenize(url))
            if tmp.token_list == None:
                print(f'{url}')
                continue
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

    # tokenize_to_file('dmoz_dir.txt', int(start_i), int(end_i))
    # tokenize_to_file('dec_xss_urls.txt', int(start_i), int(end_i))

    # tokenize_to_std('dec_xss_urls.txt', int(start_i))
    # tokenize_to_std('dmoz_dir.txt', int(start_i))

    # tokenize_to_std('dec_xss_urls.txt', 10)
    # tokenize_to_std('dmoz_dir.txt', 10)

    print(f"{longest_url('dec_xss_urls.txt__20211203-134417_0--1.dat')}")
    print(f"{longest_url('dmoz_dir.txt__20211203-134415_0--1.dat')}")

    # url = 'http://website/search.php'
    # tmp = URLTokens(url, tokenize(url))
    # print_token_list(tmp)
        