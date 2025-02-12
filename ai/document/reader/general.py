import os
import tempfile
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Set, Type, Union
from zipfile import ZipExtFile

from phi.document import Document
from phi.document.reader import Reader
from phi.document.reader.csv_reader import CSVReader
from phi.document.reader.docx import DocxReader
from phi.document.reader.json import JSONReader
from phi.document.reader.pdf import PDFReader
from phi.document.reader.text import TextReader
from pydantic import Field

from .base import BaseReader
from .excel import ExcelReader
from .image import ImageReader
from .pptx import PPTXReader
from .raw import RawReader
from .zip import ZipReader


class GeneralReader(BaseReader):
    handler: Reader = Field(default_factory=RawReader)

    def file_extension(self, file_name: str) -> str:
        if "." in file_name:
            return file_name.rsplit(".", 1)[-1].lower()
        return ""

    def _write_temp_file(self, file: IO[Any]) -> Path:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(file.read())
        temp_file.close()
        return Path(temp_file.name), temp_file

    def read(
        self,
        file: Union[str, Path, IO[Any], ZipExtFile],
        supported_extensions: Set[str] = {"*"},
    ) -> List[Document]:
        """
        Reads the given file using an appropriate reader determined by the file extension.

        Supported extensions:
          - pdf   => PDFReader
          - csv   => CSVReader
          - pptx  => PPTXReader
          - txt, md  => TextReader
          - docx  => DocxReader
          - json  => JSONReader
          - xlsx, xls  => ExcelReader
          - zip   => ZipReader
          - png, jpg, jpeg, gif, bmp, tiff, webp  => ImageReader
          - *   => RawReader

        If the extension does not match any of the above, the GenericReader is used.

        Args:
            file: The file to read.
            supported_extensions: The extensions to support; pass "*" to support all supported extensions.

        Returns:
            A list of documents.
        """
        temp_file: Optional[Path] = None
        try:
            # Determine the file extension
            ext = None

            if isinstance(file, str):
                file = Path(file)

            if isinstance(file, (Path, IO, ZipExtFile)):
                ext = self.file_extension(file.name)

            if not ext:
                raise ValueError("file extension not found!")

            if "*" not in supported_extensions:
                if ext not in supported_extensions:
                    return []

            EXT_ALIASES: Dict[str, str] = {
                "md": "txt",
                "xls": "xlsx",
                # images
                "png": "image",
                "jpg": "image",
                "jpeg": "image",
                "gif": "image",
                "bmp": "image",
                "tiff": "image",
                "webp": "image",
            }

            EXT2READER_CLASS: Type[Reader] = {
                "pdf": PDFReader,
                "csv": CSVReader,
                "pptx": PPTXReader,
                "txt": TextReader,
                "docx": DocxReader,
                "json": JSONReader,
                "xlsx": ExcelReader,
                "zip": ZipReader,
                "image": ImageReader,
            }.get(EXT_ALIASES.get(ext, ext), RawReader)

            self.handler = EXT2READER_CLASS(chunking_strategy=self.chunking_strategy)

            if ext == "json":
                if not isinstance(file, Path):
                    file, temp_file = self._write_temp_file(file)

            elif isinstance(self.handler, RawReader) and not callable(
                getattr(file, "read", None)
            ):
                raise ValueError("file has no read method")

            return self.handler.read(file)

        finally:
            if temp_file:
                # delete the temp file
                os.remove(temp_file.name)
