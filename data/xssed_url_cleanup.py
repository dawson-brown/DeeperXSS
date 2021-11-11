from urllib.parse import urlparse
from types import Tuple
from enum import Enum
from base64 import b64decode
from urllib import parse
from html import unescape
import string

class DeepURL:

    def __init__(self, url: str):
        self.parsed_url = urlparse(url)
        self.query = self.parsed_url.query
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
        pass

    def _de_utf7_decode(self, query_string : str) -> Tuple[bool, str]:
        pass
    
    def _de_url_decode(self, query_string : str) -> Tuple[bool, str]:
        if query_string.find('%') == -1:
            raise ValueError
        return parse.unquote(query_string)

    def _de_html_decode(self, query_string : str) -> Tuple[bool, str]:
        return unescape(query_string)

    def _de_dword_decode(query_string : str) -> Tuple[bool, str]:
        pass

    def _de_hex_decode(self, query_string : str) -> Tuple[bool, str]:
        return bytes.fromhex(query_string).decode("utf-8")

    def _de_octal_decode(self, query_string : str) -> Tuple[bool, str]:
        pass

    def _de_base64_decode(self, query_string : str) -> Tuple[bool, str]:
        query_string = query_string.replace('-', '=')\
                .replace('.', '+')\
                .replace('_', '/')

        return b64decode(query_string).decode("utf-8")

    def _de_decode(self, query_string : str) -> Tuple[bool, str]:
        
        all_decodings = list()
        for decoder in self._decodings:
            all_decodings.append( decoder(query_string) )

        for past, dec_str in all_decodings:
            if past:
                (next_past, next_dec_str) = self._de_decode(dec_str)
                if next_past and any(c not in string.printable for c in next_dec_str):
                    return (True, next_dec_str)
                else:
                    return (False, next_dec_str)

        if any(c not in string.printable for c in query_string):
            return (True, query_string)
        else:
            return (False, query_string)


    def decode_query(self, encoding : str) -> Tuple[bool, str]:
        
        if encoding == 'usascii':
            return self._de_us_ascii_decode(self.query)

        if encoding == 'base64':
            return self._de_base64_decode(self.query)

        if encoding == 'utf7':
            return self._de_utf7_decode(self.query)

        if encoding == 'url':
            return self._de_url_decode(self.query)

        if encoding == 'html':
            return self._de_html_decode(self.query)

        if encoding == 'dword':
            return self._de_dword_decode(self.query)

        if encoding == 'hex':
            return self._de_hex_decode(self.query)

        if encoding == 'octal':
            return self._de_octal_decode(self.query)

        return self._de_decode(self.query)


with open('xss_urls.txt', 'w') as outfile, open('xss_urls_tmp.json', 'r') as infile:
    lines = infile.read().splitlines()
    for line in lines:
        clean_line = line[9:]
        clean_line = clean_line[:-3]

        # outfile.write(clean_line)
        # outfile.write('\n')

