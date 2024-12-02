#
# A sample dataset source:
#   https://www.kaggle.com/datasets/manisha717/dataset-of-pdf-files
#
import sys
from pathlib import Path
from time import time

from phi.embedder.openai import OpenAIEmbedder
from phi.knowledge.combined import CombinedKnowledgeBase
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
from phi.utils.log import logger
from phi.vectordb.pgvector import PgVector2

from ai.agents.base import agent_settings
from db.session import db_url
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
            embedder=OpenAIEmbedder(
                model=agent_settings.embedding_model, dimensions=1536
            ),
        ),
        num_documents=5,
    )

    if len(sys.argv) != 2:
        logger.info("Usage: python pgvec.py <target_folder>")
        sys.exit(1)

    TARGET_FOLDER = Path(sys.argv[1])
    PDFs = list(TARGET_FOLDER.glob("*.pdf"))

    if not PDFs:
        logger.info(f"No PDFs found in '{TARGET_FOLDER}'")
        sys.exit(1)

    logger.info("Found `{}` pdf documents...".format(len(PDFs)))

    reader = PDFReader()

    bulk = []
    start = time()
    for index, file in enumerate(PDFs):
        try:
            bulk.extend(reader.read(file))
            logger.info(
                "[Bulk upload][{:.2f}%][Avg. Time: {:.2f} seconds] Processing...".format(
                    ((index + 1) * 100) / len(PDFs), ((time() - start) / (index + 1))
                )
            )
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
    logger.info("Bulk upload: {}".format(len(bulk)))

    for i in range(0, len(bulk), 40):
        ckb.load_documents(bulk[i : i + 40], upsert=True)
        logger.info("{:.2f}% Upserted documents".format((i + 40) * 100 / len(bulk)))

    end = time()

    logger.info("[Bulk upload][DONE]")

    dur = end - start

    logger.info(
        "[Bulk upload] Time taken: {} seconds / avg time: {:.2f} seconds.".format(
            dur, dur / len(PDFs)
        )
    )

    start = time()

    for index, file in enumerate(PDFs):
        try:
            ckb.load_documents(reader.read(file), upsert=True)
            logger.info(
                "[Single upload][{:.2f}%][Avg. Time: {:.2f} seconds] Processing...".format(
                    ((index + 1) * 100) / len(PDFs), ((time() - start) / (index + 1))
                )
            )
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")

    end = time()

    logger.info("[Single upload][DONE]")

    dur = end - start

    logger.info(
        "[Single upload] Time taken: {} seconds / avg time: {:.2f} seconds.".format(
            dur, dur / len(PDFs)
        )
    )


if __name__ == "__main__":
    main()
