# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:09:50 2018

@author: Shubham
"""

import bonobo
import requests
from bs4 import BeautifulSoup

class SampleETL:
    def generate_data(self):
        yield 'foo'
        yield 'bar'
        yield 'baz'
        
    def uppercase(self, x: str):
        return x.upper()
    
    def output(self, x: str):
        print (x)
    
    def build_graph(self):
        graph = bonobo.Graph(
                self.generate_data,
                self.uppercase,
                self.output,)
        return graph

# Using Zillow and Redfin data
class PriceETL:
    def __init__(self, zillow_url, redfin_url, headers):
        self.zillow_url = zillow_url
        self.redfin_url = redfin_url
        self.headers = headers
        
    # extract data from these websites
    def scrape_zillow(self):
        price = ''
        r = requests.get(self.zillow_url, headers=self.headers)
        if r.status_code == 200:
            html = r.text.strip()
            soup = BeautifulSoup(html, 'lxml')
            price_status_section = soup.select('.home-summary-row')
            if len(price_status_section) > 1:
                price = price_status_section[1].text.strip()
        return price
        
    def scrape_redfin(self):
        price = ''
        r = requests.get(self.redfin_url, headers=self.headers)
        if r.status_code == 200:
            html = r.text.strip()
            soup = BeautifulSoup(html, 'lxml')
            price_section = soup.find('span', {'itemprop': 'price'})
            if price_section:
                price = price_section.text.strip()
        return price
    
    def extract(self):
        yield self.scrape_zillow()
        yield self.scrape_redfin()
        
    def transform(self, price: str):
        t_price = price.replace(',', '').lstrip('$')
        return float(t_price)
        
    def load(self, price: float):
        with open('pricing.txt', 'a+', encoding='utf8') as f:
            f.write((str(price) + '\n'))
    
    
if __name__ == '__main__':
    headers = {
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
        'referrer': 'https://google.com'
    }
    zillow_url ='https://www.zillow.com/homedetails/41-Norton-Ave-Dallas-PA-18612/2119501298_zpid/'
    redfin_url = 'https://www.redfin.com/TX/Dallas/2619-Colby-St-75204/unit-B/home/32251730'
    p = PriceETL(zillow_url, redfin_url, headers)
    graph = bonobo.Graph(
        p.extract,
        p.transform,
        p.load,
    )
    bonobo.run(graph)
    