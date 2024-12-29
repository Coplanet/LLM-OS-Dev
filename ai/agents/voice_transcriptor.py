import tempfile
from textwrap import dedent
from typing import Optional, Tuple, Union

import google.generativeai as genai
from google.generativeai.types.file_types import File
from phi.agent import Agent
from phi.model.google import Gemini
from phi.workflow import RunResponse
from pydantic import BaseModel, Field

from helpers.utils import text2audio


class Transcription(BaseModel):
    transcription: str = Field(..., description="Transcription of the audio.")
    next_agent_prompt_based_on_transcription: str = Field(
        ...,
        description=(
            "Next agent's prompt based on the transcription, "
            "if no prompt is generated, return an empty string."
        ),
    )
    needs_next_agent: bool = Field(
        ...,
        description="Whether the transcription can be used as a prompt to another agent.",
    )


agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description=dedent(
        """
        You are a transcription agent that transcribes the audio and determines if the resulted transcription
        can be used as a prompt to another agent.
        """
    ).strip(),
    instructions=[
        dedent(
            """\
            Only return the transcription. If the task is purely transcription,
            do not generate a prompt for the next agent.
            """
        ).strip(),
        (
            "**VERY CRITICAL:** Keep the language of the transcription and prompt same as the "
            "original input voice if possible."
        ),
        (
            "**IMPORTANT:** If user haven't EXPLICITLY asked for TRANSCRIPTION, Make sure you "
            "generate a prompt for the next agent."
        ),
        "Next agent prompt is anything except the audio transcription.",
        "If the transcription is not clear, return an empty string for the next agent's prompt.",
        "**IMPORTANT:** Be very specific in the next agent's prompt, do not be vague.",
        "**IMPORTANT:** Don't confuse or pass your description/instructions to the next agent.",
        dedent(
            """\
            **IMPORTANT:** If the instruction is not JUST transcription and the next agent needs to perform
            but the generated instruction is not clear or vague,
            pass the next agent a prompt so he can ask for clarification.
            """
        ).strip(),
        "**CRITICAL:** Next agent cannot transcribe the audio; never ask for ANY transcription from the next agent.",
        "**CRITICAL:** Next agent's prompt **should sound** that it's comming from the user not you as an agent.",
        (
            "**CRITICAL:** Next agent's prompt **should be** exactly the same thing that user asked for don't change "
            "the question, instruction or statement or anything else."
        ),
        "**DEFINITION:** Next agent is the agent that will perform the next task based on your output.",
    ],
    response_model=Transcription,
)

__all__ = ["agent", "Transcription"]


def voice2prompt(audio_data: Union[str, bytes]) -> Tuple[bool, str, str]:
    """
    Transcribe the audio and determine if the resulted transcription
    can be used as a prompt to another agent.

    Args:
        audio_data (Union[str, bytes]):
            The audio data to transcribe.

        stream (bool):
            Whether to stream the transcription.

    Returns:
        Tuple[bool, str, str]:
            A tuple containing a boolean indicating if the transcription can be
            used as a prompt to another agent and the transcription itself.

            The boolean is True if the transcription can be used as a prompt
            and False otherwise.

            The second string is the prompt for the next agent.

            The third string is the transcription of the audio.
    """

    if isinstance(audio_data, str):
        audio_data = text2audio(audio_data)

    audio_file: Optional[File] = None

    # store audio bytes in a temp file
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
        temp_file.write(audio_data)
        temp_file_path = temp_file.name

        audio_file = genai.upload_file(temp_file_path)

    if not audio_file:
        raise Exception("Failed to upload audio file to Google Generative AI.")

    try:
        result: RunResponse = agent.run(
            dedent(
                """
                Transcribe the audio and determine if the resulted transcription
                can be used as a prompt to another agent.
                """
            ).strip(),
            audio=audio_file,
        )

        if isinstance(result, RunResponse):
            result = result.content
            if result.needs_next_agent:
                return (
                    True,
                    result.next_agent_prompt_based_on_transcription,
                    result.transcription,
                )

            return False, result.transcription, result.transcription

        return False, "", ""

    finally:
        audio_file.delete()
