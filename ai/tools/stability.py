import io
import os
from hashlib import sha256
from os import getenv
from time import time
from typing import Literal, Optional

import boto3
import requests
from phi.agent import Agent
from phi.model.content import Image
from phi.tools import Toolkit
from phi.utils.log import logger


class Stability(Toolkit):
    def __init__(
        self,
        model: Optional[Literal["ultra", "core", "sd3"]] = "ultra",
        api_key: Optional[str] = os.getenv("STABILITY_API_KEY"),
        output_format: Optional[Literal["jpeg", "png", "webp"]] = "png",
        aspect_ratio: Optional[
            Literal["16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"]
        ] = "1:1",
        s3_bucket: Optional[str] = os.getenv("AWS_S3_BUCKET", "llm-os"),
        s3_access_key: Optional[str] = os.getenv(
            "AWS_S3_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID")
        ),
        s3_secret_key: Optional[str] = os.getenv(
            "AWS_S3_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY")
        ),
        s3_region: Optional[str] = os.getenv(
            "AWS_S3_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        ),
        s3_path: Optional[str] = os.getenv("AWS_S3_STABILITY_PATH", "stability-tool"),
    ):
        super().__init__(name="stability")

        self.model = model
        self.api_key = api_key or getenv("STABILITY_API_KEY")
        self.output_format = output_format
        self.aspect_ratio = aspect_ratio
        self.s3_bucket = s3_bucket
        self.s3_region = s3_region
        self.s3_access_key = s3_access_key
        self.s3_secret_key = s3_secret_key
        self.s3_path = s3_path

        # Validations
        if model not in ["ultra", "core", "sd3"]:
            raise ValueError(
                "Invalid model. Please choose from 'ultra', 'core', or 'sd3'."
            )

        if not self.api_key:
            logger.error(
                "STABILITY_API_KEY not set. Please set the STABILITY_API_KEY environment variable."
            )

        self.register(self.create_image)

    def create_image(self, agent: Agent, prompt: str) -> str:
        """Use this function to generate an image for a prompt.

        Args:
            prompt (str): A text description of the desired image.

        Returns:
            str: A message indicating if the image has been generated successfully or an error message.
        """
        if not self.api_key:
            return "Please set the STABILITY_API_KEY"

        try:
            logger.debug(f"Generating image using prompt: {prompt}")
            response = requests.post(
                f"https://api.stability.ai/v2beta/stable-image/generate/{self.model}",
                headers={
                    "authorization": f"Bearer {self.api_key}",
                    "accept": "image/*",
                },
                files={"none": ""},
                data={
                    "prompt": prompt,
                    "output_format": self.output_format,
                    "aspect_ratio": self.aspect_ratio,
                },
            )

            file_name = "{}/{}-{}-{}.{}".format(
                self.s3_path,
                self.model,
                sha256(prompt.encode()).hexdigest(),
                time(),
                self.output_format,
            )

            if response.status_code == 200:
                # upload to AWS S3
                session = boto3.Session(
                    aws_access_key_id=self.s3_access_key,
                    aws_secret_access_key=self.s3_secret_key,
                    region_name=self.s3_region,
                )

                s3_client = session.client("s3")
                # Wrap the response content in a BytesIO object
                file_obj = io.BytesIO(response.content)
                s3_client.upload_fileobj(
                    file_obj,
                    self.s3_bucket,
                    file_name,
                )
                # get the url presigned
                url = s3_client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.s3_bucket, "Key": file_name},
                    ExpiresIn=3600 * 24 * 30,
                )

            else:
                raise Exception(str(response.json()))

            logger.debug("Image generated successfully")

            # Update the run response with the image URLs
            agent.add_image(
                Image(
                    id=file_name,
                    url=url,
                    original_prompt=prompt,
                    revised_prompt=prompt,
                )
            )
            return "Image has been generated successfully and will be displayed below"
        except Exception as e:
            logger.error(f"Failed to generate image: {e}")
            return f"Error: {e}"
