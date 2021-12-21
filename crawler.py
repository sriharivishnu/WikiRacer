import scrapy
import spacy
import time

import requests

import random as rand
from heapq import heappush, heappop

from scrapy.crawler import CrawlerProcess
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

LOG_FILE = "log.txt"
WIKIPEDIA_URL = "https://en.wikipedia.org"

START = "apple"
TARGET = "raisin"

assert requests.get(WIKIPEDIA_URL).status_code == 200
assert requests.get(WIKIPEDIA_URL + "/wiki/" + START).status_code == 200
assert requests.get(WIKIPEDIA_URL + "/wiki/" + TARGET).status_code == 200


class WikiRacer(scrapy.Spider):
    driver = webdriver.Chrome(ChromeDriverManager().install())
    name = "wikiracer"
    start_urls = [WIKIPEDIA_URL + "/wiki/" + START]
    visited = 0
    k = 1
    r_max = 0
    r_replacement = 0.05
    visited_urls = set()
    nlp = spacy.load("en_core_web_lg")
    target = nlp(TARGET)

    custom_settings = {
        "LOG_FILE": LOG_FILE,
        "LOG_STDOUT": False,
        "DEPTH_PRIORITY": 1,
        "SCHEDULER_DISK_QUEUE": "scrapy.squeues.PickleFifoDiskQueue",
        "SCHEDULER_MEMORY_QUEUE": "scrapy.squeues.FifoMemoryQueue",
    }

    def parse(self, response):
        self.driver.get(response.request.url)
        WikiRacer.visited += 1
        heap = []
        r = []
        for a in response.css(".mw-parser-output > p > a::attr(href)").extract():
            title = a.split("/")[-1]
            title = title.replace("_", " ")

            if title.lower().strip() == TARGET.lower().strip():
                print(f"Found {TARGET} at {response.request.url}")
                self.driver.get(WIKIPEDIA_URL + a)
                raise scrapy.exceptions.CloseSpider(
                    reason=f"Found link to {TARGET} at {response.request.url}"
                )

            if not all(x.isalnum() or x.isspace() for x in title):
                continue

            if a in WikiRacer.visited_urls:
                continue

            target_similarity = self.nlp(title).similarity(self.target)
            heappush(heap, (target_similarity, a))
            if len(heap) > self.k:
                t = heappop(heap)
                if self.r_max > 0 and rand.random() < self.r_replacement:
                    if len(r) + 1 > self.r_max:
                        r[rand.randint(0, len(r) - 1)] = t
                    else:
                        r.append(t)

        if r is not None:
            for y in r:
                heap.append(y)

        for item in heap:
            self.visited_urls.add(item[1])
            yield scrapy.Request(WIKIPEDIA_URL + item[1], self.parse)


process = CrawlerProcess()

process.crawl(WikiRacer)
process.start()


time.sleep(5)
WikiRacer.driver.quit()
