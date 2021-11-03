import xml

def dmoz_random_url_selection(url_list: list(str), num: int) -> list(str):
    pass


def dmoz_write_urls_to_json(url_list: list(str), filename: str) -> None:
    pass


def dmoz_urls(filename: str, filter=None) -> list(str):
    pass


def main():
    url_list = dmoz_urls('filename', '?')
    url_list = dmoz_random_url_selection(url_list, 40000)
    dmoz_write_urls_to_json(url_list, '')


if __name__ == '__main__':
    main()