import base64
from typing import IO, Any, List

from phi.document.base import Document
from phi.document.reader.base import Reader
from phi.utils.log import logger


class GenericReader(Reader):
    """Reader for any files as binary"""

    chunk: bool = False

    def read(self, path: IO[Any]) -> List[Document]:
        if not path:
            raise ValueError("No path provided")

        try:
            logger.info(f"Reading: {path}")
            file_name = path.name.split(".")[0]
            file_content = path.read()

            documents = [
                Document(
                    name=file_name,
                    id=f"{file_name}_{1}",
                    meta_data={"page": 1},
                    content=base64.b64encode(file_content),
                )
            ]
            if self.chunk:
                chunked_documents = []
                for document in documents:
                    chunked_documents.extend(self.chunk_document(document))
                return chunked_documents

            return documents

        except Exception:
            raise
