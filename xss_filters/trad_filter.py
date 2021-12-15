from bleach import clean
import re
import numpy as np
from metrics import get_precision, get_recall, get_f1, get_accuracy


def xss_sanitizer(filename: str):
    total_sanitized = 0
    total_not_sanitized = 0
    with open(filename, 'r') as not_dec_xss:
        lines = not_dec_xss.read().splitlines()
        total = len(lines)
        for line in lines:
            cleaned_line = clean(line)
            if cleaned_line == line:
                total_not_sanitized += 1
            else:
                total_sanitized += 1

    print(f'total: {total}, total sanitized: {total_sanitized}, total not sanitized: {total_not_sanitized}')


def tag_search(dmoz: str, xss: str):

    open_script = re.compile('<[^>]*>', re.IGNORECASE)
    close_script = re.compile('</[^>]*>', re.IGNORECASE)

    with open(dmoz, 'r') as dmoz_f, open(xss, 'r') as xss_f:
        lines = dmoz_f.read().splitlines()

        expected = []
        results = []
        for url in lines:
            expected.append(0)
            if len(open_script.findall(url)) > 0 or len(close_script.findall(url)) > 0:
                results.append(1)
            else:
                results.append(0)


        lines = xss_f.read().splitlines()
        for url in lines:
            expected.append(1)
            if len(open_script.findall(url)) > 0 or len(close_script.findall(url)) > 0:
                results.append(1)
            else:
                results.append(0)

        expected = np.array(expected)
        results = np.array(results)
        result = {
            'precision': get_precision(results, expected),
            'recall': get_recall(results, expected),
            'f1': get_f1(results, expected),
            'accuracy': get_accuracy(results, expected)
        }

        return result



def main():
    xss_sanitizer('data/dec_xss_urls.txt')
    xss_sanitizer('data/xss_urls.txt')


    

if __name__ == '__main__':
    main()
