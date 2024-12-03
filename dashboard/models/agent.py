from typing import Optional

from django.db import models
from phi.model.base import Model as PhiBaseModel
from phi.model.ollama import Ollama
from phi.model.openai import OpenAIChat
from phi.utils.log import logger

from ai.agents.base import Agent

from .base import BaseModel


class AIModels(models.TextChoices):
    GPT = ("gpt",)
    LLaMA = ("llama",)

    @property
    def model(self) -> PhiBaseModel:
        match self:
            case AIModels.GPT:
                return OpenAIChat
            case AIModels.LLaMA:
                return Ollama
            case _:
                raise ValueError(f"Unknown model: {self}")

    @classmethod
    def detect(cls, agent: Agent) -> "AIModels":
        if isinstance(agent.model, OpenAIChat):
            return AIModels.GPT
        elif isinstance(agent.model, Ollama):
            return AIModels.LLaMA
        else:
            raise ValueError(f"Unknown model: {agent.model}")


class APIKey(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    model = models.CharField(max_length=255, choices=AIModels.choices)
    key = models.CharField(max_length=255, unique=True)
    description = models.TextField(blank=True, null=True)

    class Meta:
        verbose_name = "API Key"
        verbose_name_plural = "API Keys"

    def __str__(self):
        return "{} [{}]".format(self.name, self.model)


class AgentConfig(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(blank=True, null=True)
    instructions = models.JSONField(blank=True, null=True, default=list)
    delegation_directives = models.JSONField(blank=True, null=True)
    model_type = models.CharField(
        max_length=255, choices=AIModels.choices, default=AIModels.GPT
    )
    model_id = models.CharField(max_length=255)
    model_config = models.JSONField(null=False, blank=False, default=dict)
    agent_config = models.JSONField(null=True, blank=True, default=dict)
    api_key = models.ForeignKey(APIKey, on_delete=models.PROTECT, null=True, blank=True)
    last_activated_at = models.DateTimeField(auto_now=True)
    enabled = models.BooleanField(default=True)

    @classmethod
    def register_or_load(
        cls, agent: Agent, model_config: dict, agent_config: dict
    ) -> Optional[Agent]:
        obj, created = cls.objects.get_or_create(
            name=agent_config.pop("name", None) or agent.name,
        )
        description = agent_config.pop("description", None)
        instructions = agent_config.pop("instructions", None)
        delegation_directives = agent_config.pop("delegation_directives", None)

        if created:
            obj.description = description or agent.description
            obj.instructions = instructions or agent.instructions
            obj.delegation_directives = (
                delegation_directives or agent.delegation_directives
            )
            obj.model_type = AIModels.detect(agent)
            obj.model_id = agent.model.id
            obj.model_config = model_config
            obj.agent_config = agent_config

        obj.save()

        if isinstance(obj.model_type, str):
            obj.model_type = AIModels(obj.model_type)

        if obj.api_key is not None:
            obj.model_config["api_key"] = obj.api_key.key

        kwargs = {
            "name": obj.name,
            "description": obj.description,
            "instructions": obj.instructions,
            "delegation_directives": obj.delegation_directives,
            "model": obj.model_type.model(**obj.model_config),
            "enabled": obj.enabled,
        }

        # Remove keys from agent_config that are already in kwargs
        for key in kwargs:
            if key in obj.agent_config:
                del obj.agent_config[key]

        # Update kwargs with remaining agent_config
        kwargs.update(obj.agent_config)

        # Add fields from agent that are not in kwargs
        for key, value in vars(agent).items():
            if key not in kwargs:
                kwargs[key] = value

        agent_ = agent.__class__(**kwargs)

        logger.info(
            "Agent `%s` loaded with `%s` model with id `%s`",
            agent_.name,
            type(agent_.model).__name__,
            agent_.model.id,
        )
        logger.info(
            'Agent `%s` loaded with descrption: "%s"', agent_.name, agent_.description
        )
        logger.info(
            "Agent `%s` loaded with instructions: %s", agent_.name, agent_.instructions
        )
        logger.info(
            "Agent `%s` loaded with delegation directives: %s",
            agent_.name,
            agent_.delegation_directives,
        )

        return agent_

    def save(self, *args, **kwargs):
        if self.agent_config is None:
            self.agent_config = {}

        if self.api_key is None:
            if "api_key" in self.model_config:
                self.api_key = APIKey.objects.filter(
                    key=self.model_config["api_key"]
                ).first()

                if self.api_key is None:
                    self.api_key, created = APIKey.objects.get_or_create(
                        name=f"{self.model_type} API Key",
                        model=self.model_type,
                        key=self.model_config["api_key"],
                    )
                    if created:
                        self.api_key.description = (
                            f"Auto generated API Key for {self.model_type} model"
                        )
                        self.api_key.save()

            if self.api_key is None:
                self.api_key = APIKey.objects.filter(model=self.model_type).first()

        if self.api_key is not None and "api_key" in self.model_config:
            del self.model_config["api_key"]

        return super().save(*args, **kwargs)

    class Meta:
        verbose_name = "Agent Config"
        verbose_name_plural = "Agent Configs"

    def __str__(self):
        return "{} [{}]{}".format(
            self.name, self.model_type, "" if self.enabled else " (disabled)"
        )
