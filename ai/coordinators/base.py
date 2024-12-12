from ai.agents.base import Agent, AgentTeam


class Coordinator(Agent):
    @classmethod
    def build(
        cls,
        team: AgentTeam,
        *args,
        **kwargs,
    ):
        kwargs["team"] = team
        if "additional_context" not in kwargs:
            kwargs["additional_context"] = ""

        if isinstance(kwargs["additional_context"], list):
            kwargs["additional_context"] = "\n".join(kwargs["additional_context"])

        if len(team) and kwargs["additional_context"]:
            kwargs["additional_context"] += "\n"

        kwargs["additional_context"] += team.delegation_directives

        return cls(*args, **kwargs)
