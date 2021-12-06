from bleach import clean
import re


def xss_filter(filename: str):
    total_filtered = 0
    total_not_filtered = 0
    with open(filename, 'r') as not_dec_xss:
        lines = not_dec_xss.read().splitlines()
        total = len(lines)
        for line in lines:
            cleaned_line = clean(line)
            if cleaned_line == line:
                total_not_filtered += 1
            else:
                # print(f'line: {line}')
                # print(f'cleaned line: {cleaned_line}')
                total_filtered += 1

    print(f'total: {total}, total filtered: {total_filtered}, total not filtered: {total_not_filtered}')
    print(f'Precision: {total_filtered / total}\n')


def tag_search(filename: str):
    total_filtered = 0
    total_not_filtered = 0
    total = 0

    with open(filename, 'r') as not_dec_xss:
        lines = not_dec_xss.read().splitlines()
        total = len(lines)
        open_script = re.compile('<[^>]*>', re.IGNORECASE)
        close_script = re.compile('</[^>]*>', re.IGNORECASE)

        for url in lines:
            # t = open_script.findall(url)
            # v = close_script.findall(url)
            if len(open_script.findall(url)) > 0 or len(close_script.findall(url)) > 0:
                total_filtered+=1
            else:
                # print(f'{url}')
                total_not_filtered+=1

    print(f'total: {total}, total filtered: {total_filtered}, total not filtered: {total_not_filtered}')
    print(f'Precision: {total_filtered / total}\n')



def main():

    # print('xss_filter - Not decoded malicious:')
    # xss_filter('data/xss_urls.txt')

    # print('\nxss_filter - Decoded malicious:')
    # xss_filter('data/dec_xss_urls.txt')

    # print('\nxss_filter - Benign:')
    # xss_filter('data/dmoz_dir.txt')


    print('\ntag_search - Not decoded malicious:')
    tag_search('data/xss_urls.txt')

    print('\ntag_search - Decoded malicious:')
    tag_search('data/dec_xss_urls.txt')

    print('\ntag_search - Benign:')
    tag_search('data/dmoz_dir.txt')


    

if __name__ == '__main__':
    main()
