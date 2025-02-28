from typing import Dict
from urllib.parse import urljoin, urlparse

import httpx
from phi.document.chunking.recursive import RecursiveChunking
from phi.document.chunking.strategy import ChunkingStrategy
from phi.document.reader.website import WebsiteReader as WR
from phi.utils.log import logger
from pydantic import Field

try:
    from bs4 import BeautifulSoup  # noqa: F401
except ImportError:
    raise ImportError(
        "The `bs4` package is not installed. Please install it via `pip install beautifulsoup4`."
    )


class WebsiteReader(WR):
    chunk: bool = True
    chunking_strategy: ChunkingStrategy = Field(default_factory=RecursiveChunking)

    def crawl(self, url: str, starting_depth: int = 1) -> Dict[str, str]:
        """
        Crawls a website and returns a dictionary of URLs and their corresponding content.

        Parameters:
        - url (str): The starting URL to begin the crawl.
        - starting_depth (int, optional): The starting depth level for the crawl. Defaults to 1.

        Returns:
        - Dict[str, str]: A dictionary where each key is a URL and the corresponding value is the main
                          content extracted from that URL.

        Note:
        The function focuses on extracting the main content by prioritizing content inside common HTML tags
        like `<article>`, `<main>`, and `<div>` with class names such as "content", "main-content", etc.
        The crawler will also respect the `max_depth` attribute of the WebCrawler class, ensuring it does not
        crawl deeper than the specified depth.
        """
        num_links = 0
        crawler_result: Dict[str, str] = {}
        primary_domain = self._get_primary_domain(url)
        # Add starting URL with its depth to the global list
        self._urls_to_crawl.append((url, starting_depth))
        while self._urls_to_crawl:
            # Unpack URL and depth from the global list
            current_url, current_depth = self._urls_to_crawl.pop(0)

            # Skip if
            # - URL is already visited
            # - does not end with the primary domain,
            # - exceeds max depth
            # - exceeds max links
            if (
                current_url in self._visited
                or not urlparse(current_url).netloc.endswith(primary_domain)
                or current_depth > self.max_depth
                or num_links >= self.max_links
            ):
                continue

            self._visited.add(current_url)
            self.delay()

            try:
                logger.debug(f"Crawling: {current_url}")
                response = httpx.get(
                    current_url,
                    timeout=10,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko; compatible; Googlebot/2.1; "
                            "+http://www.google.com/bot.html) Chrome/W.X.Y.Z Safari/537.36"
                        )
                    },
                )
                main_content = None
                try:
                    soup = BeautifulSoup(response.content, "html.parser")

                    # Extract main content
                    main_content = self._extract_main_content(soup)

                    if not main_content:
                        main_content = soup.get_text()

                except Exception as e:
                    logger.debug(f"Failed to extract main content: {e}")

                if not main_content:
                    main_content = response.content

                if main_content:
                    crawler_result[current_url] = main_content
                    num_links += 1

                # Add found URLs to the global list, with incremented depth
                for link in soup.find_all("a", href=True):
                    full_url = urljoin(current_url, link["href"])
                    parsed_url = urlparse(full_url)
                    if parsed_url.netloc.endswith(primary_domain) and not any(
                        parsed_url.path.endswith(ext)
                        for ext in [".pdf", ".jpg", ".png"]
                    ):
                        if (
                            full_url not in self._visited
                            and (full_url, current_depth + 1) not in self._urls_to_crawl
                        ):
                            self._urls_to_crawl.append((full_url, current_depth + 1))

            except Exception as e:
                logger.debug(f"Failed to crawl: {current_url}: {e}")
                pass

        return crawler_result
