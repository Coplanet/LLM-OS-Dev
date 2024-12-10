from pathlib import Path
from typing import Optional

from phi.tools.file import FileTools


class FileIOTools(FileTools):
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        save_files: bool = True,
        read_files: bool = True,
        list_files: bool = True,
    ):
        super().__init__(base_dir, save_files, read_files, list_files)
        self.name = "file_io_tools"
