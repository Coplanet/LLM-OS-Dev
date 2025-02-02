#
# A sample dataset source:
#   https://www.kaggle.com/datasets/manisha717/dataset-of-pdf-files
#
import sys
from pathlib import Path
from time import time

from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.combined import CombinedKnowledgeBase
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.pgvector import PgVector2

from db.session import db_url
from helpers.log import logger
from workspace.settings import extra_settings


def main():
    ckb = CombinedKnowledgeBase(
        sources=[
            PDFKnowledgeBase(
                path=extra_settings.knowledgebase_dir, reader=PDFReader(chunk=True)
            )
        ],
        vector_db=PgVector2(
            db_url=db_url,
            collection="pgvec_benchmark",
            embedder=OpenAIEmbedder(dimensions=1536),
        ),
        num_documents=5,
    )

    if len(sys.argv) != 2:
        logger.debug("Usage: python pgvec.py <target_folder>")
        sys.exit(1)

    TARGET_FOLDER = Path(sys.argv[1])
    PDFs = list(TARGET_FOLDER.glob("*.pdf"))

    if not PDFs:
        logger.debug(f"No PDFs found in '{TARGET_FOLDER}'")
        sys.exit(1)

    logger.debug("Found `{}` pdf documents...".format(len(PDFs)))

    reader = PDFReader()

    bulk = []
    start = time()
    for index, file in enumerate(PDFs):
        try:
            bulk.extend(reader.read(file))
            logger.debug(
                "[Bulk upload][{:.2f}%][Avg. Time: {:.2f} seconds] Processing...".format(
                    ((index + 1) * 100) / len(PDFs), ((time() - start) / (index + 1))
                )
            )
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
    logger.debug("Bulk upload: {}".format(len(bulk)))

    for i in range(0, len(bulk), 40):
        ckb.load_documents(bulk[i : i + 40], upsert=True)
        logger.debug("{:.2f}% Upserted documents".format((i + 40) * 100 / len(bulk)))

    end = time()

    logger.debug("[Bulk upload][DONE]")

    dur = end - start

    logger.debug(
        "[Bulk upload] Time taken: {} seconds / avg time: {:.2f} seconds.".format(
            dur, dur / len(PDFs)
        )
    )

    start = time()

    for index, file in enumerate(PDFs):
        try:
            ckb.load_documents(reader.read(file), upsert=True)
            logger.debug(
                "[Single upload][{:.2f}%][Avg. Time: {:.2f} seconds] Processing...".format(
                    ((index + 1) * 100) / len(PDFs), ((time() - start) / (index + 1))
                )
            )
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")

    end = time()

    logger.debug("[Single upload][DONE]")

    dur = end - start

    logger.debug(
        "[Single upload] Time taken: {} seconds / avg time: {:.2f} seconds.".format(
            dur, dur / len(PDFs)
        )
    )


if __name__ == "__main__":
    main()
