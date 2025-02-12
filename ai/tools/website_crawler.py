import json
from typing import List, Optional

from phi.document.base import Document
from phi.tools.website import WebsiteKnowledgeBase, WebsiteTools
from phi.utils.log import logger

from ai.document.reader.website import WebsiteReader


class WebSiteCrawlerTools(WebsiteTools):
    def __init__(self, knowledge_base: Optional[WebsiteKnowledgeBase] = None):
        super().__init__(knowledge_base)
        self.name = "website_crawler_tools"

    def read_url(self, url: str) -> str:
        """This function reads a url and returns the content.

        :param url: The url of the website to read.
        :return: Relevant documents from the website.
        """
        website = WebsiteReader()

        logger.debug(f"Reading website: {url}")
        relevant_docs: List[Document] = website.read(url=url)
        return json.dumps([doc.to_dict() for doc in relevant_docs])
