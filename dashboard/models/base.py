# from llmdj.externalize import activate_django

# activate_django()

from django.db import models


class BaseModel(models.Model):
    id = models.BigAutoField(primary_key=True)

    meta = models.JSONField(blank=True, null=True, default=dict)

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True
