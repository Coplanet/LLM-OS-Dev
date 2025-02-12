from pathlib import Path
from typing import IO, Any, List, Union

import pandas as pd
from phi.document.base import Document

from helpers.log import logger

from .base import BaseReader


class ExcelReader(BaseReader):
    """Reader for Excel files"""

    chunk: bool = False

    def read(self, path: Union[str, Path, IO[Any]]) -> List[Document]:
        if not path:
            raise ValueError("No path provided")

        try:
            logger.debug(f"Reading: {path}")
            splits = path.name.split(".")
            file_name = splits[0]
            file_extension = splits[-1]
            # Choose the engine based on the file extension
            if file_extension == "xls":
                engine = "xlrd"
            elif file_extension == "xlsx":
                engine = "openpyxl"
            else:
                raise ValueError(
                    "Unsupported file extension: {}".format(file_extension)
                )

            # Read all sheets from the Excel file
            all_sheets = pd.read_excel(path, sheet_name=None, engine=engine)

            # Convert each sheet to a dictionary with records orientation
            json_contents = {
                sheet_name: df.to_dict(orient="records")
                for sheet_name, df in all_sheets.items()
            }
            documents = []

            for sheet, contents in json_contents.items():
                if isinstance(contents, dict):
                    contents = [contents]
                for content in contents:
                    documents.append(
                        Document(
                            name=file_name,
                            id=f"{file_name}_{sheet}",
                            meta_data={"sheet_name": sheet},
                            content=str(content),
                        )
                    )

            if self.chunk:
                logger.debug("Chunking documents not yet supported for ExcelReader")

            return documents
        except Exception:
            raise
