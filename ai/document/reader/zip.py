import zipfile
from pathlib import Path
from typing import IO, Any, List, Set, Union

from phi.document.base import Document

from helpers.log import logger

from .base import BaseReader


class ZipReader(BaseReader):
    """Reader for ZIP files"""

    def read(
        self,
        file: Union[str, Path, IO[Any]],
        encoding: str = "utf-8",
        supported_extensions: Set[str] = {"*"},
    ) -> List[Document]:
        """
        Reads a zip file and returns a list of Documents.
        Each file within the zip (excluding directories) is converted into a Document.

        Parameters:
            file (Union[Path, IO[Any]]): A Path object or a file-like object opened in binary mode.
            encoding (str): The encoding to decode file contents (default: 'utf-8').

        Returns:
            List[Document]: A list of Document objects.
        """
        if not file:
            raise ValueError("No file provided")

        documents: List[Document] = []
        try:
            if isinstance(file, str):
                file = Path(file)
            # Check if the file is a Path object or a file-like object.
            if isinstance(file, Path):
                if not file.exists():
                    raise FileNotFoundError(f"Could not find file: {file}")
                logger.info(f"Reading zip file from path: {file}")
                with file.open("rb") as file_obj:
                    with zipfile.ZipFile(file_obj) as zf:
                        documents = self._extract_documents_from_zip(
                            zf, encoding, supported_extensions
                        )
            else:
                logger.info(
                    f"Reading uploaded zip file: {getattr(file, 'name', 'unknown')}"
                )
                file.seek(0)
                with zipfile.ZipFile(file) as zf:
                    documents = self._extract_documents_from_zip(
                        zf, encoding, supported_extensions
                    )

            return documents

        except Exception as e:
            logger.error(f"Error reading zip file: {getattr(file, 'name', file)}: {e}")
            return []

    def _extract_documents_from_zip(
        self, zf: zipfile.ZipFile, encoding: str, supported_extensions: Set[str]
    ) -> List[Document]:
        """
        Helper method to extract all files from a ZipFile object and convert them to Documents.

        Parameters:
            zf (zipfile.ZipFile): An opened ZipFile object.
            encoding (str): The encoding to use when decoding file contents.

        Returns:
            List[Document]: A list of Document objects created from the files in the zip.
        """
        from ai.document.reader.general import GeneralReader

        documents: List[Document] = []
        reader = GeneralReader(chunking_strategy=self.chunking_strategy)

        for info in zf.infolist():
            # Skip directories
            if info.is_dir():
                continue

            try:
                with zf.open(info) as f:
                    documents.extend(
                        reader.read(f, supported_extensions=supported_extensions)
                    )

            except UnicodeDecodeError as e:
                logger.error(f"Error decoding file {info.filename}: {e}")
                continue

        chunked_documents: List[Document] = []
        if self.chunk:
            for doc in documents:
                chunked_docs = self.chunk_document(doc)
                chunked_documents.extend(chunked_docs)

        else:
            chunked_documents = documents

        return chunked_documents
