import io
import json
import os
from hashlib import sha256
from os import getenv
from time import sleep, time
from typing import Literal, Optional, Tuple, Union

import boto3
import requests
import streamlit as st
from phi.agent import Agent
from phi.model.content import Image
from phi.model.message import Message
from phi.tools import Toolkit
from phi.utils.log import logger
from sqlalchemy import orm

from db.session import get_db_context
from db.tables import UserBinaryData, UserNextOp


class Stability(Toolkit):
    DEFAULT_NEGATIVE_PROMPT = ", ".join(
        [
            "bad anatomy",
            "bad hands",
            "three hands",
            "three legs",
            "bad arms",
            "missing legs",
            "missing arms",
            "poorly drawn face",
            "poorly rendered hands",
            "bad face",
            "fused face",
            "cloned face",
            "worst face",
            "three crus",
            "extra crus",
            "fused crus",
            "worst feet",
            "three feet",
            "fused feet",
            "fused thigh",
            "three thigh",
            "extra thigh",
            "worst thigh",
            "missing fingers",
            "extra fingers",
            "ugly fingers",
            "long fingers",
            "bad composition",
            "horn",
            "extra eyes",
            "huge eyes",
            "2girl",
            "amputation",
            "disconnected limbs",
            "cartoon",
            "cg",
            "3d",
            "unreal",
            "animate",
            "cgi",
            "render",
            "artwork",
            "illustration",
            "3d render",
            "cinema 4d",
            "artstation",
            "octane render",
            "mutated body parts",
            "painting",
            "oil painting",
            "2d",
            "sketch",
            "bad photography",
            "bad photo",
            "deviant art",
            "aberrations",
            "abstract",
            "anime",
            "black and white",
            "collapsed",
            "conjoined",
            "creative",
            "drawing",
            "extra windows",
            "harsh lighting",
            "jpeg artifacts",
            "low saturation",
            "monochrome",
            "multiple levels",
            "overexposed",
            "oversaturated",
            "photoshop",
            "rotten",
            "surreal",
            "twisted",
            "UI",
            "underexposed",
            "unnatural",
            "unreal engine",
            "unrealistic",
            "video game",
            "deformed body features",
        ]
    )

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
        self.register(self.add_feature_or_change_accurately)
        self.register(self.search_and_recolor)

    def _req(
        self,
        url: str,
        data: dict = {},
        files: dict = {"none": ""},
    ) -> requests.Response:
        start = time()
        try:
            if "output_format" not in data:
                data["output_format"] = self.output_format

            if "aspect_ratio" not in data:
                data["aspect_ratio"] = self.aspect_ratio

            return requests.post(
                url,
                headers={
                    "authorization": f"Bearer {self.api_key}",
                    "accept": "image/*",
                },
                files=files,
                data=data,
            )

        finally:
            end = time()
            logger.debug(
                f"Time taken for '{self.__class__.__name__}.{self._req.__name__}': {end - start} seconds."
            )

    def _store_in_s3(
        self,
        agent: Agent,
        response: requests.Response,
        original_prompt: str = "",
        revised_prompt: str = "",
        **include_in_name,
    ) -> str:
        start = time()
        try:
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
                    file_obj,
                    self.s3_bucket,
                    file_name,
                    ExtraArgs={"ACL": "public-read"},
                )
                url = "https://{}.s3.{}.amazonaws.com/{}".format(
                    self.s3_bucket,
                    s3_client.meta.region_name,
                    file_name,
                )
                # we need to wait for the image to be ready in the S3 bucket
                while requests.head(url).status_code != 200:
                    sleep(0.3)

                with get_db_context() as db:
                    image = UserBinaryData.save_data(
                        db,
                        agent.session_id,
                        UserBinaryData.IMAGE,
                        UserBinaryData.UPSTREAM,
                        response.content,
                    )
                    st.session_state["selected_image"] = image.id

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

        finally:
            end = time()
            logger.debug(
                f"Time taken for '{self.__class__.__name__}.{self._store_in_s3.__name__}': {end - start} seconds"
            )

    @classmethod
    def latest_user_image_with_mask(
        cls, agent: Agent
    ) -> Optional[Tuple[UserBinaryData, Optional[UserBinaryData]]]:
        start = time()
        try:
            with get_db_context() as db:
                image = cls.latest_user_image(
                    agent, type=UserBinaryData.IMAGE, return_data=False, db=db
                )
                mask = cls.latest_user_image(
                    agent, type=UserBinaryData.IMAGE_MASK, return_data=False, db=db
                )
                return image, mask

        finally:
            end = time()
            logger.debug(
                f"Time taken for '{cls.__name__}.{cls.latest_user_image_with_mask.__name__}': {end - start} seconds"
            )

    @classmethod
    def latest_user_image(
        cls,
        agent: Agent,
        type: Optional[Literal["image", "image_mask"]] = UserBinaryData.IMAGE,
        return_data: Optional[bool] = True,
        db: Optional[orm.Session] = None,
    ) -> Optional[Union[bytes, UserBinaryData]]:
        start = time()
        try:
            if (
                type == UserBinaryData.IMAGE
                and "selected_image" in st.session_state
                and isinstance(st.session_state["selected_image"], int)
            ):
                image_id = st.session_state["selected_image"]
                with get_db_context() as db:
                    image = UserBinaryData.get_by_id(
                        db,
                        agent.session_id,
                        image_id,
                    )
                    if image:
                        return image.data if return_data else image

            def fetch_image(dbi: orm.Session):
                image = UserBinaryData.get_data(
                    dbi,
                    agent.session_id,
                    type,
                ).first()
                if image:
                    return image.data if return_data else image
                return None

            if db:
                return fetch_image(db)

            with get_db_context() as db:
                return fetch_image(db)

        finally:
            end = time()
            logger.debug(
                f"Time taken for '{cls.__name__}.{cls.latest_user_image.__name__}': {end - start} seconds."
            )

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
        image = self.latest_user_image(agent)

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
        seed: int,
        grow_mask: Optional[int] = 3,
        negative_prompt: Optional[str] = None,
    ) -> str:
        """Use this function to search for a prompt and replace it with the new prompt.
        for the same prompt don't send parallel requests. it will be handled by the toolkit.
        it will parse the latest user images and edit them. Don't use it to remove background.
        **CRITICAL NOTE**: Don't use this function when the user needs to select/mask/highlight some area in the image.

        Args:
            prompt (str): The new image's prompt to replace.
            search_prompt (str): The prompt to search objects or areas in the image.
            negative_prompt (str): A blurb of text describing what you do not wish to see in the output image.
            grow_mask (int): Grows the edges of the mask outward in all directions by the specified number of pixels. \
                The expanded area around the mask will be blurred, which can help smooth the transition between \
                inpainted content and the original image.
            seed (int): A specific value to guide the randomness of the generation.

        Returns:
            str: A message indicating if the image has been generated successfully or an error message.
        """
        image = self.latest_user_image(agent)

        if not image:
            return "No image found"

        response = self._req(
            "https://api.stability.ai/v2beta/stable-image/edit/search-and-replace",
            files={"image": image},
            data={
                "prompt": prompt,
                "search_prompt": search_prompt,
                "grow_mask": grow_mask,
                "negative_prompt": "{}, {}".format(
                    self.DEFAULT_NEGATIVE_PROMPT, negative_prompt
                ),
                "seed": seed,
            },
        )

        self._store_in_s3(agent, response, prompt, f"{prompt} -> {search_prompt}")

        return "Image has been edited successfully and will be displayed below"

    def add_feature_or_change_accurately(
        self,
        agent: Agent,
        prompt: str,
        seed: int,
        thing_to_avoid_when_editing: Optional[str] = None,
    ) -> str:
        """Add features to an image while preserving its original content.
        **CRITICAL NOTE**: **BUT DON'T MAKE UP THING BY YOURSELF ON BEHALF OF THE USER**.
        **CRITICAL NOTE**: The `thing_to_avoid_when_editing` should be a very descriptive \
            text for the undesired image outcome, the negative prompt should be send as \
            possitive words, e.g. instead of "no crippled face", send "crippled face".
        **CRITICAL NOTE**: Use this function when the user needs to select/mask/highlight some area \
            in the image for accurate editing.

        Args:
            prompt (str): User's exact input prompt.
            thing_to_avoid_when_editing (str): Text describing the elements to avoid when editing the image.
            seed (int): Base on the input prompt and the image's description, choose a seed value at your choice.

        Returns:
            str: Message indicating success or error.
        """
        image, mask = self.latest_user_image_with_mask(agent)

        if not image:
            return "No image found"

        if not mask:
            with get_db_context() as db:
                nextop = UserNextOp.save_op(
                    db,
                    agent.session_id,
                    UserNextOp.GET_IMAGE_MASK,
                    {"image_id": image.id},
                )

            return json.dumps(
                {
                    "message": {"mask": "required"},
                    "next_step": True,
                    "user_additional_output": "Please provide a mask for the image",
                    "stop_calling_other_tools": True,
                    "add_to_memory": True,
                    "method": "add_feature",
                    "class": "stability",
                    "nextop_id": nextop.id,
                }
            )

        response = self._req(
            "https://api.stability.ai/v2beta/stable-image/edit/inpaint",
            files={"image": image.data, "mask": mask.data},
            data={
                "prompt": prompt,
                "negative_prompt": "{}, {}".format(
                    self.DEFAULT_NEGATIVE_PROMPT, thing_to_avoid_when_editing
                ),
                "seed": seed,
            },
        )

        if response.status_code == 200:
            with get_db_context() as db:
                UserNextOp.delete_all_by_key(
                    db,
                    agent.session_id,
                    UserNextOp.EDIT_IMAGE_USING_MASK,
                    auto_commit=False,
                )
                UserNextOp.delete_all_by_key(
                    db,
                    agent.session_id,
                    UserNextOp.GET_IMAGE_MASK,
                    auto_commit=False,
                )
                UserBinaryData.delete_all_by_type(
                    db, agent.session_id, UserBinaryData.IMAGE_MASK
                )

        self._store_in_s3(agent, response, prompt, f"{prompt}")

        return "Image has been edited successfully and will be displayed below"

    def remove_background(self, agent: Agent) -> str:
        """Use this function to Remove Background accurately segments the foreground
        from an image and implements and removes the background.

        Returns:
            str: A message indicating if the image has been generated successfully or an error message.
        """
        image = self.latest_user_image(agent)

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

    def search_and_recolor(
        self,
        agent: Agent,
        prompt: str,
        select_prompt: str,
        seed: int,
        thing_to_avoid_when_editing: Optional[str] = None,
    ) -> str:
        """
        Use this function to The Search and Recolor service provides the ability to change the color
        of a specific object in an image using a prompt. This service is a specific version of inpainting
        that does not require a mask. The Search and Recolor service will automatically segment the object
        and recolor it using the colors requested in the prompt.

        Args:
            prompt (str): What you wish to see in the output image. A strong, descriptive prompt that \
                clearly defines elements, colors, and subjects will lead to better results.
            select_prompt (str): Short description of what to search for in the image.
            thing_to_avoid_when_editing (str): A blurb of text describing what you do not wish to see \
                in the output image.
            seed (int): A specific value to guide the randomness of the generation.

        Returns:
            str: A message indicating if the image has been edited successfully or an error message.
        """
        image = self.latest_user_image(agent)

        if not image:
            return "No image found"

        response = self._req(
            "https://api.stability.ai/v2beta/stable-image/edit/search-and-recolor",
            files={"image": image},
            data={
                "prompt": prompt,
                "select_prompt": select_prompt,
                "negative_prompt": "{}, {}".format(
                    self.DEFAULT_NEGATIVE_PROMPT, thing_to_avoid_when_editing
                ),
                "seed": seed,
            },
        )

        self._store_in_s3(agent, response, prompt, f"{prompt} -> {select_prompt}")

        return "Image has been edited successfully and will be displayed below"
