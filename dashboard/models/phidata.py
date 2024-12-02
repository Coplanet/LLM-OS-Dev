# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
# from llmdj.externalize import activate_django

# activate_django()

from django.db import models


class AgentSessions(models.Model):
    session_id = models.CharField(primary_key=True)
    agent_id = models.CharField(blank=True, null=True)
    user_id = models.CharField(blank=True, null=True)
    memory = models.JSONField(blank=True, null=True)
    agent_data = models.JSONField(blank=True, null=True)
    user_data = models.JSONField(blank=True, null=True)
    session_data = models.JSONField(blank=True, null=True)
    created_at = models.BigIntegerField(blank=True, null=True)
    updated_at = models.BigIntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = "agent_sessions"


class AlembicVersion(models.Model):
    version_num = models.CharField(primary_key=True, max_length=32)

    class Meta:
        managed = False
        db_table = "alembic_version"


class LlmOsDocuments(models.Model):
    id = models.CharField(primary_key=True)
    name = models.CharField(blank=True, null=True)
    meta_data = models.JSONField(blank=True, null=True)
    content = models.TextField(blank=True, null=True)
    embedding = models.TextField(blank=True, null=True)  # This field type is a guess.
    usage = models.JSONField(blank=True, null=True)
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)
    content_hash = models.CharField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = "llm_os_documents"
