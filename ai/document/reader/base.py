from phi.document.chunking.recursive import RecursiveChunking
from phi.document.chunking.strategy import ChunkingStrategy
from phi.document.reader.base import Reader
from pydantic import Field


class BaseReader(Reader):
    chunk: bool = True
    chunking_strategy: ChunkingStrategy = Field(default_factory=RecursiveChunking)
