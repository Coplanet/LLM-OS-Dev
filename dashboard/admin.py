from django.contrib import admin

# Register your models here.
from .models import (
    AgentConfig,
    AgentSessions,
    AlembicVersion,
    APIKey,
    Config,
    LlmOsDocuments,
    UserConfig,
)

admin.site.register(Config)
admin.site.register(APIKey)
admin.site.register(UserConfig)
admin.site.register(AgentConfig)
admin.site.register(AgentSessions)
admin.site.register(AlembicVersion)
admin.site.register(LlmOsDocuments)
