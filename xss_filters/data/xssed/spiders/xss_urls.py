import scrapy
from bs4 import BeautifulSoup

class XssUrlsSpider(scrapy.Spider):
    name = 'xss_urls'
    start_urls = ['http://www.xssed.com/archive/page=1']

    def parse(self, response):

        for mirror in response.xpath('//th[@id="tableborder"]//tr/th/a[@target="_blank"]/@href').getall():   
            yield response.follow(mirror, self.parse_url)

        next_page = response.xpath('//th[@id="contentpaneOpen"]/table/div/span[@class="activelink"]/following::a[1]/@href').get()
        if next_page is not None and next_page != '/':
            yield response.follow(next_page, self.parse)

    def parse_url(self, response):
        url = response.xpath('//th[@id="contentpaneOpen"]/table//tr/th')[9].get()
        th = BeautifulSoup(url)
        url_str = th.get_text()[5:]
        
        yield {
            'url': url_str
        }

