from typing import Optional

from phi.tools.website import WebsiteKnowledgeBase, WebsiteTools


class WebSiteCrawlerTools(WebsiteTools):
    def __init__(self, knowledge_base: Optional[WebsiteKnowledgeBase] = None):
        super().__init__(knowledge_base)
        self.name = "website_crawler_tools"
