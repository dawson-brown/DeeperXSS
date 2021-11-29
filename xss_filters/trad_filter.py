from bleach import clean


def xss_filter(filename: str):
    print(filename)
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

def main():
    xss_filter('../data/xss_urls.txt')
    xss_filter('../data/dec_xss_urls.txt')
    xss_filter('../data/dmoz_dir.txt')

    

if __name__ == '__main__':
    main()
