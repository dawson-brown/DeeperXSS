# DeeperXSS

## DATA
The Xssed.com scraper can be found [here](https://github.com/dawson-brown/DeeperXSS/blob/main/xss_filters/data/xssed/spiders/xss_urls.py). The Dmoz data was downloaded as zip file from [here](https://curlz.org/dmoz_rdf/content.rdf.u8.gz).


Full list of Dmoz benign URLs: [[benign](https://github.com/dawson-brown/DeeperXSS/blob/main/xss_filters/data/dmoz_dir.txt)](https://github.com/dawson-brown/DeeperXSS/blob/main/xss_filters/data/dmoz_dir.txt)

Full list of none-decoded Xssed URLs: [https://github.com/dawson-brown/DeeperXSS/blob/main/xss_filters/data/xss_urls.txt](https://github.com/dawson-brown/DeeperXSS/blob/main/xss_filters/data/xss_urls.txt)

Full list of decoded Xssed URLs: [https://github.com/dawson-brown/DeeperXSS/blob/main/xss_filters/data/dec_xss_urls.txt](https://github.com/dawson-brown/DeeperXSS/blob/main/xss_filters/data/dec_xss_urls.txt)


### Decoder
The custom decoder can be found [here](https://github.com/dawson-brown/DeeperXSS/blob/519b92cb04870fdd31339caaa47a2bcd8f4da66d/xss_filters/data/xssed_url_clean.py#L67).

### Tokenizer
The custom tokenizer can be found [here](https://github.com/dawson-brown/DeeperXSS/blob/519b92cb04870fdd31339caaa47a2bcd8f4da66d/xss_filters/data/tokenizer.py#L68).

### Word2Vec

The code for training the CBOW model can be found [here](https://github.com/dawson-brown/DeeperXSS/blob/main/xss_filters/word2vec.py).

### Model Training
The code for training the models:
[Primary Model](https://github.com/dawson-brown/DeeperXSS/blob/main/xss_filters/lstm_softmax.py)
[Sigmoid Output](https://github.com/dawson-brown/DeeperXSS/blob/main/xss_filters/lstm_sigmoid.py)
[No Embedding Layer](https://github.com/dawson-brown/DeeperXSS/blob/main/xss_filters/lstm_sequence.py)
[Randomly initialized Embedding Layer](https://github.com/dawson-brown/DeeperXSS/blob/main/xss_filters/lstm_random_embed.py)
