from urllib.parse import urlparse, urlunparse
from typing import Tuple
from base64 import b64decode
from urllib import parse
from html import unescape

class DeepURL:

    def __init__(self, url: str = None):
        self.url = None


        if url is not None:
            self.url = urlparse(url)

        self._decodings = [
            self._de_url_decode,
            self._de_html_decode,
            self._de_hex_decode,
            self._de_base64_decode,
            self._de_url_unicode_decode
        ]
    
    def _de_url_unicode_decode(self, query_string : str) -> Tuple[bool, str]:
        try:
            try_dec = query_string.replace('%', '\\').encode('latin-1').decode('unicode-escape')
            if try_dec == query_string:
                return (False, query_string)        
            return (True, try_dec)
        except:
            return (False, query_string)

    def _de_url_decode(self, query_string : str) -> Tuple[bool, str]:
        try_dec = parse.unquote_plus(query_string)
        if query_string == try_dec:
            return (False, query_string)
        return (True, try_dec)


    def _de_html_decode(self, query_string : str) -> Tuple[bool, str]:
        dec_str = unescape(query_string)
        if dec_str == query_string:
            return (False, query_string)
        return (True, dec_str)


    def _de_hex_decode(self, query_string : str) -> Tuple[bool, str]:
        try:
            dec_str = bytes.fromhex(query_string).decode("utf-8")
            return (True, dec_str)
        except:
            return (False, query_string)


    def _de_base64_decode(self, query_string : str) -> Tuple[bool, str]:
        query_string = query_string.replace('-', '=')\
                .replace('.', '+')\
                .replace('_', '/')

        try:
            dec_str = b64decode(query_string, True).decode("utf-8")
            return (True, dec_str)
        except:
            return (False, query_string)


    def _de_decode(self, query_string : str) -> Tuple[bool, str]:

        if query_string == '':
            return (False, query_string)

        all_decodings = list()
        for decoder in self._decodings:
            all_decodings.append( decoder(query_string) )

        some_decoded = False
        for past, dec_str in all_decodings:
            some_decoded = some_decoded or past
            if past:
                (next_past, next_dec_str) = self._de_decode(dec_str)
                if next_past:
                    return (True, next_dec_str)

        if some_decoded:
            return (False, query_string)
        else:
            return (True, query_string)


    def decode_url(self) -> Tuple[bool, str]:
        
        dec_q, dec_query = self._de_decode(self.url.query)   
        dec_p, dec_path = self._de_decode(self.url.path)
        deq_pa, dec_params = self._de_decode(self.url.params)
        deq_f, dec_frag = self._de_decode(self.url.fragment)

        tmp_url = urlunparse([self.url.scheme, 
            'website',
            dec_path,
            dec_params,
            dec_query,
            dec_frag])
        tmp_url = tmp_url.replace('\n', ' ').replace('\r', '')

        self.url = urlparse(tmp_url)
        return ( dec_q or dec_p or deq_pa or deq_f,  \
            tmp_url )


with open('dec_xss_urls.txt', 'w') as dec_outfile,\
    open('n_dec_xss_urls.txt', 'w') as n_dec_outfile,\
    open('xss_urls.json', 'r') as infile:
    lines = infile.read().splitlines()
    decode_strings = set()

    for i, line in enumerate(lines):
        clean_line = line[9:]
        clean_line = clean_line[:-3]

        url = DeepURL(url=clean_line)
        decoded, decoded_str = url.decode_url()

        if decoded:
            decode_strings.add(decoded_str)
        else:
            n_dec_outfile.write(clean_line)
            n_dec_outfile.write('\n')

    for dec_str in decode_strings:
        dec_outfile.write(dec_str)
        dec_outfile.write('\n')

