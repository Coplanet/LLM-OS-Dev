from django.db import models

from .base import BaseModel


class Config(BaseModel):
    key = models.CharField(max_length=255, null=False, blank=False, unique=True)
    value = models.TextField(null=False, blank=False)

    @classmethod
    def get(cls, key, default=None):
        try:
            return cls.objects.get(key=key).value

        except cls.DoesNotExist:
            return default

    def __str__(self):
        return self.key
