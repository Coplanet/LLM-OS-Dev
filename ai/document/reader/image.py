import base64
from io import BytesIO
from pathlib import Path
from typing import IO, Any, List, Union

from phi.document.base import Document
from phi.document.reader.base import Reader
from PIL import Image

from helpers.log import logger


class ImageReader(Reader):
    """Reader for image files"""

    chunk: bool = False

    def read(self, path: Union[str, Path, IO[Any]]) -> List[Document]:
        if not path:
            raise ValueError("No path provided")

        try:
            logger.debug(f"Reading: {path}")
            img_name = path.name.split(".")[0]

            image = Image.open(path)
            buffer = BytesIO()
            image.save(buffer, format="webp")
            encoding = base64.b64encode(buffer.getvalue()).decode("utf-8")
            content = f"data:image/webp;base64,{encoding}"

            documents = [
                Document(
                    name=img_name,
                    id=f"{img_name}_{1}",
                    meta_data={"page": 1},
                    content=content,
                )
            ]
            if self.chunk:
                logger.debug("Chunking documents not yet supported for ImageReader")

            return documents
        except Exception:
            raise
