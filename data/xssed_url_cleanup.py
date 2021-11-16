from urllib.parse import urldefrag, urljoin, urlparse, urlunparse, urlunsplit
from typing import Tuple
from base64 import b64decode
from urllib import parse
from html import unescape
import string

class DeepURL:

    def __init__(self, url: str = None):
        self.url = None


        if url is not None:
            self.url = urlparse(url)

        self._decodings = [
            self._de_us_ascii_decode,
            self._de_utf7_decode,
            self._de_url_decode,
            self._de_html_decode,
            self._de_dword_decode,
            self._de_hex_decode,
            self._de_base64_decode
        ]
    
    
    def _de_us_ascii_decode(self, query_string : str) -> Tuple[bool, str]:
        return (False, query_string)


    def _de_utf7_decode(self, query_string : str) -> Tuple[bool, str]:
        return (False, query_string)
    

    def _de_url_decode(self, query_string : str) -> Tuple[bool, str]:
        try_dec = parse.unquote(query_string)
        if query_string == try_dec:
            return (False, query_string)
        return (True, try_dec)


    def _de_html_decode(self, query_string : str) -> Tuple[bool, str]:
        dec_str = unescape(query_string)
        if dec_str == query_string:
            return (False, query_string)
        return (True, dec_str)


    def _de_dword_decode(self, query_string : str) -> Tuple[bool, str]:
        return (False, query_string)


    def _de_hex_decode(self, query_string : str) -> Tuple[bool, str]:
        try:
            dec_str = bytes.fromhex(query_string).decode("utf-8")
            return (True, dec_str)
        except:
            return (False, query_string)


    def _de_octal_decode(self, query_string : str) -> Tuple[bool, str]:
        return (False, query_string)


    def _de_base64_decode(self, query_string : str) -> Tuple[bool, str]:
        query_string = query_string.replace('-', '=')\
                .replace('.', '+')\
                .replace('_', '/')

        try:
            dec_str = b64decode(query_string).decode("utf-8")
            return (True, dec_str)
        except:
            return (False, query_string)


    def _de_decode(self, query_string : str) -> Tuple[bool, str]:

        all_decodings = list()
        for decoder in self._decodings:
            all_decodings.append( decoder(query_string) )

        for past, dec_str in all_decodings:
            if past:
                (next_past, next_dec_str) = self._de_decode(dec_str)
                if next_past:
                    return (True, next_dec_str)

        if any(c not in string.printable for c in query_string):
            return (False, query_string)
        else:
            return (True, query_string)


    def decode_url(self) -> str:
        
        if self.url.query == '':
            dec_q, dec_query = (False, None)
        else:
            dec_q, dec_query = self._de_decode(self.url.query)
        
        if self.url.path == '':
            dec_p, dec_path = (False, None)
        else:
            dec_p, dec_path = self._de_decode(self.url.path)

        if self.url.params == '':
            deq_pa, dec_params = (False, None)
        else:
            deq_pa, dec_params = self._de_decode(self.url.params)


        if self.url.params == '':
            deq_f, dec_frag = (False, None)
        else:
            deq_f, dec_frag = self._de_decode(self.url.fragment)
#<scheme>://<netloc>/<path>;<params>?<query>#<fragment>
        tmp_url = urlunparse([self.url.scheme, 
            self.url.netloc,
            dec_path,
            dec_params,
            dec_query,
            dec_frag])

        self.url = urlparse(tmp_url)
        return tmp_url


with open('xss_urls.txt', 'w') as outfile, open('xss_urls_tmp.json', 'r') as infile:
    lines = infile.read().splitlines()
    for line in lines:
        clean_line = line[9:]
        clean_line = clean_line[:-3]

        url = DeepURL(url=clean_line)
        decoded = url.decode_url()

        # print(f'{decoded}')
        outfile.write(decoded)
        outfile.write('\n')

