# DeeperXSS

## Decoder
The decoder recursively tries decoding until ascii JS is left. The support encodings are:

1. US-ASCII (7-bit)
2. UTF-7
3. URL
4. DWORD
5. HEX
6. OCTAL
7. BASE64

Multi-layered encoding is supported, but mixed encoding is not.