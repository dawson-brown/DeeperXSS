from pickle import dump, load

def load_tokenized_urls(filename: str):

    print('Pickle Load:\n')
    with open(filename, "rb") as f:
        while True:
            try:
                yield load(f)
            except EOFError:
                break


def main():
    
    for tokenized_url in load_tokenized_urls(''):
        for token in tokenized_url:
            pass

    

if __name__ == '__main__':
    main()