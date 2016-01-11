from bs4 import BeautifulSoup
import requests

def createScrapper(url):
        html = requests.get(url).text
        return BeautifulSoup(html, "html5lib")

