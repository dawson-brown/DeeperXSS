from urllib.parse import urlparse
from types import Tuple
from enum import Enum
from base64 import b64decode
from urllib import parse
from html import unescape

class DeepURL:

    def __init__(self, url: str):
        self.parsed_url = urlparse(url)
        self.query = self.parsed_url.query
    
    def _de_us_ascii_decode(self, string : str) -> Tuple[bool, str]:
        pass

    def _de_utf7_decode(self, string : str) -> Tuple[bool, str]:
        pass
    
    def _de_url_decode(self, string : str) -> Tuple[bool, str]:
        if string.find('%') == -1:
            raise ValueError
        return parse.unquote(string)

    def _de_html_decode(self, string : str) -> Tuple[bool, str]:
        return unescape(string)

    def _de_dword_decode(string : str) -> Tuple[bool, str]:
        pass

    def _de_hex_decode(self, string : str) -> Tuple[bool, str]:
        return bytes.fromhex(string).decode("utf-8")

    def _de_octal_decode(self, string : str) -> Tuple[bool, str]:
        pass

    def _de_base64_decode(self, string : str) -> Tuple[bool, str]:
        string = string.replace('-', '=')\
                .replace('.', '+')\
                .replace('_', '/')

        return b64decode(string).decode("utf-8")

    def _de_decode(self, string : str) -> Tuple[bool, str]:
        
        # put decode methods in array--loop over (makes easier to add to)
        (usascii, usascii_dec) = self._de_us_ascii_decode(string)
        (base64, base64_dec) = self._de_base64_decode(string)
        (octal, octal_dec) = self._de_octal_decode(string)
        (hex, hex_dec) = self._de_hex_decode(string)
        (dword, dword_dec) = self._de_dword_decode(string)
        (html, html_dec) = self._de_html_decode(string)
        (url, url_dec) = self._de_url_decode(string)
        (utf7, utf7_dec) = self._de_utf7_decode(string)

        if usascii:
            (usascii, usascii_dec) = self._de_decode(usascii_dec)
        if base64:
            (base64, base64_dec) = self._de_decode(base64_dec)
        if octal:
            (octal, octal_dec) = self._de_decode(octal_dec)
        if hex:
            (hex, hex_dec) = self._de_decode(hex_dec)
        if dword:
            (dword, dword_dec) = self._de_decode(dword_dec)
        if html:
            (html, html_dec) = self._de_decode(html_dec)
        if url:
            (url, url_dec) = self._de_decode(url_dec)
        if utf7:
            (utf7, utf7_dec) = self._de_decode(utf7_dec)

        return (True, string)


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

