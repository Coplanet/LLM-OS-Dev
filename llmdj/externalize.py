import os

import django

DJANGO_ACTIVATED = False


# Set the Django settings module
def activate_django():
    global DJANGO_ACTIVATED

    if not DJANGO_ACTIVATED:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "llmdj.settings")
        django.setup()
        DJANGO_ACTIVATED = True


__all__ = ["activate_django"]
