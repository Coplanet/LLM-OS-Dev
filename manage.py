#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

from llmdj.externalize import DJANGO_ACTIVATED  # noqa


def main():
    """Run administrative tasks."""
    global DJANGO_ACTIVATED

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "llmdj.settings")

    DJANGO_ACTIVATED = True

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
