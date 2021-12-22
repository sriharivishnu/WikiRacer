import logging
import time
import configparser
import random as rand
from urllib.parse import unquote

import scrapy
import spacy
import requests


from difflib import SequenceMatcher
from typing import List, Optional, Set
from attr import dataclass
from heapq import heappush, heappop

from scrapy.crawler import CrawlerProcess
from selenium import webdriver
from spacy.language import Language

try:
    from selenium.webdriver.chrome.webdriver import WebDriver
    from webdriver_manager.chrome import ChromeDriverManager

except ImportError:
    logging.getLogger("WikiRacer").warn(
        "Could not import selenium.webdriver.chrome.webdriver webdriver_manager. Chrome driver will not be downloaded."
    )


@dataclass
class Config:
    """
    k: max number of links to follow on page.
    This takes the top k similar links and follows them
    A large k may impact performance since more links are followed,
    but could also improve since it has multiple possible paths to follow
    """

    k: int = 1

    """
    max number of random links to follow per page
    if r_max is 0, then no random links will be followed
    This introduces noise into the search space, and
    can help to escape page loops. However, performance
    is impacted with random links
    """
    r_max: int = 0

    """
    The probability of a link to be followed
    if randomly chosen
    """
    r_replacement: float = 0.005

    """
    Can override Wikipedia domain to support
    other languages.
    """
    WIKIPEDIA_DOMAIN: str = "en.wikipedia.org"

    """
    Flag to indicate whether to open a chrome browser
    to visually see the path taken by the bot
    """
    UI: bool = False

    """
    Which NLP model to use for spacy. The model
    must already be installed on the system
    """
    NLP_MODEL: str = "en_core_web_lg"

    """
    The CSS selector used to select links from a web page. Can be
    overridden for custom functionality
    """
    CSS_LINK_EXTRACTOR: str = ".mw-parser-output > p > a::attr(href)"

    """
    File to store logs
    """
    LOG_FILE: str = "log.txt"

    def load(self):
        """
        Loads the configuration from a file
        """
        config = configparser.ConfigParser()
        config.read("config.ini")

        self.k = config["DEFAULT"].getint("k", self.k)
        self.r_max = config["DEFAULT"].getint("r_max", self.r_max)
        self.r_replacement = config["DEFAULT"].getboolean(
            "r_replacement", self.r_replacement
        )
        self.WIKIPEDIA_DOMAIN = config["DEFAULT"].get(
            "wikipedia_domain", self.WIKIPEDIA_DOMAIN
        )
        self.WIKIPEDIA_URL = f"https://{self.WIKIPEDIA_DOMAIN}"

        self.UI = config["DEFAULT"].getboolean("ui", self.UI)
        self.NLP_MODEL = config["DEFAULT"].get("nlp_model", self.NLP_MODEL)
        self.CSS_LINK_EXTRACTOR = config["DEFAULT"].get(
            "css_link_extractor", self.CSS_LINK_EXTRACTOR
        )
        self.LOG_FILE = config["DEFAULT"].get("log_file", self.LOG_FILE)
        return self


class WikiRacer(scrapy.Spider):
    """
    name of the spider
    """

    name = "WikiRacer"

    """
    UI driver to open chrome browser to
    watch the bot in action
    """
    driver: Optional[WebDriver] = None

    """
    initialize the spider with the start URL
    """
    start_urls: List[str] = []

    """
    counter to keep track of visited URLs
    """
    visited: int = 0

    """
    stores the urls that have been visited
    this is needed in addition to the built-in visited set
    since we need to calculate the max similarity only for links
    that have not been visited yet
    
    currently, this stores something like "/wiki/Banana" as an entry
    """
    visited_urls: Set[str] = set()

    """
    Used to initialize the nlp object. Used mainly to calculate 
    the similarity between two words.
    """
    nlp: Language = None

    """
    Store the target token in the class.
    """
    target: Language = None

    """
    Settings for the spider.
    Set the log file, and make the search BFS instead of DFS.
    When following multiple related links on the same page, BFS
    makes much more sense than DFS since when we reach target the 
    first time, that will be the "shortest" distance.
    """
    custom_settings = {
        "LOG_FILE": "log.txt",
        "LOG_STDOUT": False,
        "DEPTH_PRIORITY": 1,
        "SCHEDULER_DISK_QUEUE": "scrapy.squeues.PickleFifoDiskQueue",
        "SCHEDULER_MEMORY_QUEUE": "scrapy.squeues.FifoMemoryQueue",
    }

    found: bool = False

    def __init__(self, start: str, target: str, config: Config = Config(), **kwargs):
        super().__init__(self.name, **kwargs)
        self.START = start
        self.TARGET = target
        self.start_urls = [config.WIKIPEDIA_URL + "/wiki/" + self.START]
        self.allowed_domains = [config.WIKIPEDIA_DOMAIN]

        self.config = config
        self.nlp = spacy.load(config.NLP_MODEL)

        # remove special characters + URL encoding
        t = target.replace("_", " ")
        t = unquote(t)

        temp_t = []
        for c in t:
            if c.isalnum() or c == " ":
                temp_t.append(c)
        t = "".join(temp_t)

        self.target_token = self.nlp(t)

        self.custom_settings["LOG_FILE"] = config.LOG_FILE

        if config.UI:
            self.driver = webdriver.Chrome(ChromeDriverManager().install())

    def parse(self, response):
        if WikiRacer.found:
            return

        print("Processing: " + response.request.url.split("/")[-1])

        # navigate to the current spider
        if hasattr(self, "driver") and self.driver is not None:
            self.driver.get(response.request.url)

        self.visited += 1

        # maintain min heap of k most similar words
        heap = []

        # randomly chosen links if applicable
        r = []

        # get all links according to css selector
        for a in response.css(self.config.CSS_LINK_EXTRACTOR).extract():
            if not a.startswith("/wiki/"):
                continue

            raw_title = a.split("/")[-1]

            # check if we have reached the target
            if raw_title.lower().strip() == self.TARGET.lower().strip():
                print(f"Found {self.TARGET} at {response.request.url}")

                if hasattr(self, "driver") and self.driver is not None:
                    self.driver.get(self.config.WIKIPEDIA_URL + a)

                WikiRacer.found = True

                # Reached destination. We can stop crawling.
                raise scrapy.exceptions.CloseSpider(
                    reason=f"Found link to {self.TARGET} at {response.request.url}"
                )

            # check if we have already visited this link before
            if a in WikiRacer.visited_urls:
                continue

            if any(x in "?#&" for x in a):
                continue

            clean_title = raw_title.replace("_", " ")
            clean_title = unquote(clean_title)

            # compute similarity between the current link and the target
            if self.target_token and self.target_token.vector_norm:
                target_similarity = self.nlp(clean_title).similarity(self.target_token)
            else:
                target_similarity = SequenceMatcher(
                    None, raw_title, self.TARGET
                ).ratio()

            # push the similarity + link into the heap
            heappush(heap, (target_similarity, a))

            # check if we have reached the max number of links to follow
            if len(heap) > self.config.k:
                # remove the minimum similarity in heap
                # this will be the least similar link
                t = heappop(heap)

                # randomly select dropped links to follow
                if self.config.r_max > 0 and rand.random() < self.config.r_replacement:
                    # maintain list of max r_max random links
                    if len(r) + 1 > self.config.r_max:
                        # replace a link at random
                        r[rand.randint(0, len(r) - 1)] = t
                    else:
                        r.append(t)

        # add the randomly chosen links
        for y in r:
            heap.append(y)

        # follow all links in the heap. These are the
        # k most similar links to the target plus
        # any random links that were added
        for item in heap:
            self.visited_urls.add(item[1])
            yield scrapy.Request(self.config.WIKIPEDIA_URL + item[1], self.parse)


if __name__ == "__main__":

    logger = logging.getLogger("WikiRacer")
    logger.info("Reading config from config.ini")

    # load the config from config.ini
    config = Config().load()

    START = input("Enter the input word. Please make sure that the Wiki page exists: ")
    TARGET = input(
        "Enter the target word. Please make sure that the Wiki page exists: "
    )

    # make sure start and end endpoints are reachable
    if requests.get(config.WIKIPEDIA_URL + "/wiki/" + START).status_code != 200:
        logger.error(
            f"The start endpoint {config.WIKIPEDIA_URL + '/wiki/' + START} is not reachable."
        )
        exit(1)

    if requests.get(config.WIKIPEDIA_URL + "/wiki/" + TARGET).status_code != 200:
        logger.error(
            f"The end endpoint {config.WIKIPEDIA_URL + '/wiki/' + TARGET} is not reachable."
        )
        exit(1)

    WikiRacer.custom_settings["LOG_FILE"] = config.LOG_FILE

    # start crawling
    start_time = time.time()
    logger.info(f"Starting crawler with settings: {config}...")
    process = CrawlerProcess(settings=WikiRacer.custom_settings)
    process.crawl(WikiRacer, start=START, target=TARGET, config=config)
    process.start()

    elapsed_time = time.time() - start_time
    logger.info(f"Crawler finished in {elapsed_time} seconds.")

    if not WikiRacer.found:
        print("Could not find the target word.")

    else:
        print(f"Successfully found the target word in {elapsed_time} seconds.")

        # sleep for a few seconds to see the final web page :)
        # before closing it
        time.sleep(3)

    # close the web driver if needed
    if WikiRacer.driver:
        WikiRacer.driver.quit()
