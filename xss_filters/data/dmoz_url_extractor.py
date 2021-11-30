from lxml import etree as ElementTree
from typing import List, Mapping
import subprocess
from math import gcd
import random

def dmoz_random_url_selection(url_list: List[str], num: int) -> List[str]:
    random.seed()
    step = random.randint(1, len(url_list)-1)
    while gcd(step, len(url_list)) != 1:
        step = random.randint(1, len(url_list)-1)

    i = random.randint(0, len(url_list)-1)
    total = 0
    url_selection = list()
    while total < num:
        url_selection.append( url_list[i] )
        total+=1
        i = (i+step) % len(url_list)

    return url_selection



def dmoz_write_urls_to_file(url_list: List[str], filename: str) -> None:
    with open(filename, 'w') as outfile:
        outfile.write('\n'.join(url_list))


def dmoz_urls_with_params(filename: str) -> List[str]:

    cmd = rf"grep \<link {filename} | grep \?"
    links_xml = subprocess.check_output(cmd,shell=True)\
        .decode("utf8")\
        .strip()\
        .split("\n")
    links = list()

    parser = ElementTree.XMLParser(recover=True)    
    for link in links_xml:
        elem = ElementTree.fromstring(link, parser)
        links.append(elem.attrib['r:resource'])

    return links



def main():
    url_list = dmoz_urls_with_params('dmoz_dir.xml')
    url_list = dmoz_random_url_selection(url_list, 40000)
    dmoz_write_urls_to_file(url_list, 'dmoz_dir.txt')


if __name__ == '__main__':
    main()