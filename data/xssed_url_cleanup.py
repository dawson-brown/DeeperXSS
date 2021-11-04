

with open('xss_urls.txt', 'w') as outfile, open('xss_urls.json', 'r') as infile:
    lines = infile.read().splitlines()
    for line in lines:
        clean_line = line[9:]
        clean_line = clean_line[:-3]
        outfile.write(clean_line)
        outfile.write('\n')