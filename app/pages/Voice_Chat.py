import streamlit as st
from audio_recorder_streamlit import audio_recorder
from phi.agent.agent import Agent
from phi.model.openai import OpenAIChat

from ai.agents.voice_transcriptor import voice2prompt
from helpers.utils import text2audio

# Initialize the agent
agent = Agent(
    model=OpenAIChat(
        id="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
    ),
    instructions=[
        "Always respond to the user in voice format.",
        "Your input is in text format. but your response should be in voice format.",
        "**CRITICAL:** Your response should be in voice format. Do not respond in text format.",
    ],
    description="You are a helpful assistant. You are able to understand and respond to the user's voice input.",
    markdown=True,
)

# Streamlit page configuration
st.set_page_config(page_title="Voice Interaction with Agent", page_icon="🎙️")

st.title("Voice Interaction with Agent")
st.markdown("Talk to the agent using your voice and receive a voice response.")

st.markdown(
    "<style>.st-key-response_audio { display: None; }</style>", unsafe_allow_html=True
)

# Define sample rate
AUDIO_SAMPLE_RATE = 44_100

# Add an audio recorder for voice messages
audio_bytes = audio_recorder(
    text="Press to Record",
    icon_size="2x",
    pause_threshold=5,
    sample_rate=AUDIO_SAMPLE_RATE,
    key="voice_input_recorder",
)

# Process the audio input
if audio_bytes:
    # reject audio with less than 2 seconds
    if len(audio_bytes) < 2 * 4 * AUDIO_SAMPLE_RATE:
        AUDIO_ERROR = st.error("Recording cannot be less than 2 seconds!", icon="⚠")
    else:
        with open("/tmp/audio.wav", "wb") as f:
            f.write(audio_bytes)

        prompt: str = ""

        with st.spinner("Agent is listening..."):
            is_prompt, prompt, transcription = voice2prompt(audio_bytes)

            if not is_prompt:
                prompt = "Read the following text and respond to it: " + transcription

        # Get response from the agent
        with st.spinner("Agent is thinking..."):
            response = agent.run(prompt)

        # Convert response text to audio
        if (
            agent.run_response.response_audio is not None
            and "data" in agent.run_response.response_audio
        ):
            with st.container(key="response_audio"):
                st.audio(
                    text2audio(agent.run_response.response_audio["data"]),
                    format="audio/wav",
                    autoplay=True,
                )
        else:
            st.error("Agent's response is not an audio response")
