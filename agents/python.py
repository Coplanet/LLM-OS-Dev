from .base import CitexGPT4Agent
from .settings import ws_settings

agent = CitexGPT4Agent(
    name="Python Agent",
    role="Advanced Python Code Writer and Executor",
    pip_install=True,
    description=(
        "The Python Agent is a specialized tool designed to not only write "
        "advanced professional Python code but also execute it seamlessly. "
        "It supports various charting libraries, including Pandas, matplotlib, numpy, seaborn, "
        "Streamlit, to facilitate data visualization and analysis."
    ),
    charting_libraries=["streamlit", "plotly", "matplotlib", "seaborn"],
    base_dir=ws_settings.scratch_dir,
    delegation_directives=[
        "To write and run Python code, delegate the task to the `Python Agent`."
    ],
)

__all__ = ["agent"]
