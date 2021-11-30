from pickle import dump, load
from data.tokenizer import URLTokens, JSToken 

def load_tokenized_urls(filename: str) -> URLTokens:

    with open(filename, "rb") as f:
        while True:
            try:
                yield load(f)
            except EOFError:
                break


def main():
    
    # loop for all bening DMOZ urls
    for tokenized_url in load_tokenized_urls('data/dmoz_dir.txt_0--1.dat'):
        for token in tokenized_url.token_list:
            value = token.value
            tok_type = token.type
            print(f'{value}, {tok_type}')

    # loop for first 500 malicious urls
    for tokenized_url in load_tokenized_urls('data/dec_xss_urls.txt_0-500.dat'):
        for token in tokenized_url.token_list:
            value = token.value
            tok_type = token.type
            print(f'{value}, {tok_type}')

    

if __name__ == '__main__':
    main()
