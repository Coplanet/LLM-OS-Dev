import base64
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
from phi.model.message import Message
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
        self.register(self.search_and_replace)
        self.register(self.outpaint)
        self.register(self.remove_background)

    def _req(
        self,
        url: str,
        data: dict = {},
        files: dict = {"none": ""},
    ) -> requests.Response:
        if "output_format" not in data:
            data["output_format"] = self.output_format

        if "aspect_ratio" not in data:
            data["aspect_ratio"] = self.aspect_ratio

        return requests.post(
            url,
            headers={"authorization": f"Bearer {self.api_key}", "accept": "image/*"},
            files=files,
            data=data,
        )

    def _store_in_s3(
        self,
        agent: Agent,
        response: requests.Response,
        original_prompt: str = "",
        revised_prompt: str = "",
        **include_in_name,
    ) -> str:
        if "original_prompt" not in include_in_name:
            include_in_name["original_prompt"] = original_prompt

        if "revised_prompt" not in include_in_name:
            include_in_name["revised_prompt"] = revised_prompt

        if not original_prompt:
            original_prompt = include_in_name["original_prompt"]

        if not revised_prompt:
            revised_prompt = include_in_name["revised_prompt"]

        if not revised_prompt:
            revised_prompt = original_prompt

        file_name = "{}/{}-{}-{}.{}".format(
            self.s3_path,
            self.model,
            sha256(str(include_in_name).encode()).hexdigest(),
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
                file_obj, self.s3_bucket, file_name, ExtraArgs={"ACL": "public-read"}
            )
            url = "https://{}.s3.{}.amazonaws.com/{}".format(
                self.s3_bucket,
                s3_client.meta.region_name,
                file_name,
            )

        else:
            raise Exception(str(response.json()))

        logger.debug("Image generated successfully")

        # Update the run response with the image URLs
        agent.add_image(
            Image(
                id=file_name,
                url=url,
                original_prompt=original_prompt,
                revised_prompt=revised_prompt,
            )
        )
        agent.memory.add_message(
            Message(
                role="tool",
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                        "image_caption": revised_prompt,
                    }
                ],
            )
        )
        agent.write_to_storage()
        return "Image has been generated successfully and will be displayed below"

    def _latest_user_image(self, agent: Agent) -> Optional[bytes]:
        for index in list(range(len(agent.run_response.messages) - 1, -1, -1)):
            if agent.run_response.messages[index].images:
                return base64.b64decode(
                    agent.run_response.messages[index].images[-1].split("base64,")[1]
                )

        for index in list(range(len(agent.memory.messages) - 1, -1, -1)):
            if isinstance(agent.memory.messages[index].content, list):
                for content in agent.memory.messages[index].content:
                    if (
                        "type" in content
                        and content["type"] == "image_url"
                        and "image_url" in content
                        and isinstance(content["image_url"], dict)
                        and "url" in content["image_url"]
                        and isinstance(content["image_url"]["url"], str)
                    ):
                        image = content["image_url"]["url"]
                        resp = requests.get(image)
                        if resp.status_code == 200:
                            return resp.content

        return None

    def outpaint(
        self,
        agent: Agent,
        prompt: Optional[str] = None,
        creativity: Optional[float] = 0.5,
        top: Optional[int] = 0,
        left: Optional[int] = 0,
        bottom: Optional[int] = 0,
        right: Optional[int] = 0,
    ) -> str:
        """
        Use this function to outpaint an image.
        This function Inserts additional content in an image to fill in the space in any direction.
        Compared to other automated or manual attempts to expand the content in an image,
        the Outpaint service should minimize artifacts and signs that the original image has been edited.

        Args:
            prompt (str): What you wish to see in the output image. A strong, descriptive prompt \
                that clearly defines elements, colors, and subjects will lead to better results.
            creativity (float): A value between 0 and 1 that controls the likelihood of creating \
                additional details not heavily conditioned by the initial image.
            top (int): The number of pixels to outpaint on the top side.
            left (int): The number of pixels to outpaint on the left side.
            bottom (int): The number of pixels to outpaint on the bottom side.
            right (int): The number of pixels to outpaint on the right side.

        Returns:
            str: A message indicating the result.
        """
        image = self._latest_user_image(agent)

        if not image:
            return "No image found"

        response = self._req(
            "https://api.stability.ai/v2beta/stable-image/edit/outpaint",
            files={"image": image},
            data={
                "left": left,
                "down": bottom,
                "right": right,
                "up": top,
                "prompt": prompt,
                "creativity": creativity,
            },
        )

        self._store_in_s3(agent, response, f"outpaint {left} {bottom}")

        return "Image has been edited successfully and will be displayed below"

    def search_and_replace(
        self,
        agent: Agent,
        prompt: str,
        search_prompt: str,
        grow_mask: Optional[int] = 3,
        negative_prompt: Optional[str] = None,
    ) -> str:
        """Use this function to search for a prompt and replace it with the new prompt.
        for the same prompt don't send parallel requests. it will be handled by the toolkit.
        it will parse the latest user images and edit them. Don't use it to remove background.

        Args:
            prompt (str): The new image's prompt to replace.
            search_prompt (str): The prompt to search objects or areas in the image.
            negative_prompt (str): A blurb of text describing what you do not wish to see in the output image.
            grow_mask (int): Grows the edges of the mask outward in all directions by the specified number of pixels. \
                The expanded area around the mask will be blurred, which can help smooth the transition between \
                inpainted content and the original image.

        Returns:
            str: A message indicating if the image has been generated successfully or an error message.
        """
        image = self._latest_user_image(agent)

        if not image:
            return "No image found"

        response = self._req(
            "https://api.stability.ai/v2beta/stable-image/edit/search-and-replace",
            files={"image": image},
            data={
                "prompt": prompt,
                "search_prompt": search_prompt,
                "grow_mask": grow_mask,
                "negative_prompt": negative_prompt,
            },
        )

        self._store_in_s3(agent, response, prompt, f"{prompt} -> {search_prompt}")

        return "Image has been edited successfully and will be displayed below"

    def remove_background(self, agent: Agent) -> str:
        """Use this function to Remove Background accurately segments the foreground
        from an image and implements and removes the background.

        Returns:
            str: A message indicating if the image has been generated successfully or an error message.
        """
        image = self._latest_user_image(agent)

        if not image:
            return "No image found"

        response = self._req(
            "https://api.stability.ai/v2beta/stable-image/edit/remove-background",
            files={"image": image},
        )

        self._store_in_s3(agent, response, "remove background", "remove background")

        return "Image has been edited successfully and will be displayed below"

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

            response = self._req(
                f"https://api.stability.ai/v2beta/stable-image/generate/{self.model}",
                data={
                    "prompt": prompt,
                },
            )

            return self._store_in_s3(agent, response, prompt)

        except Exception as e:
            logger.error(f"Failed to generate image: {e}")
            return f"Error: {e}"
