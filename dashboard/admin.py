from django.contrib import admin

# Register your models here.
from .models import AgentConfig, AgentSessions, AlembicVersion, APIKey, LlmOsDocuments

admin.site.register(APIKey)
admin.site.register(AgentConfig)
admin.site.register(AgentSessions)
admin.site.register(AlembicVersion)
admin.site.register(LlmOsDocuments)
