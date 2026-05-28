"""The utility module for defining some simple utility functions, which is directly used in IO, and so on."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rich import style
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

DEFAULT_PRINT_SETTINGS = {"width": 120}


@dataclass
class PrintConfig:
    """The configuration for print settings, which is used to control the print output.

    Args:
        verbosity: int = 0
            the current verbosity level, which is used to control the print output
        settings: dict
            the style settings for each verbosity level
    """

    verbosity: int = 0
    settings: dict = field(default_factory=lambda: DEFAULT_PRINT_SETTINGS.copy())

    def update(self, **kwargs) -> None:
        """Update the print settings with the given keyword arguments."""
        self.settings.update(kwargs)

    def reset(self) -> None:
        """Reset settings to default."""
        self.settings = DEFAULT_PRINT_SETTINGS.copy()


# Module-level singleton, shared across the module
_config = PrintConfig()


def get_config() -> PrintConfig:
    """Get the current print configuration."""
    return _config


def get_console_settings() -> dict:
    """Return console keyword arguments derived from the global config."""
    return _config.settings.copy()


def update_print_settings(**kwargs) -> None:
    """Update the global print settings."""
    _config.update(**kwargs)


class Printer:
    """
    Args:
        STANDARD: int = 0
            the level of print of STANDARD
        DEBUG: int = 1
            the level of print of DEBUG, which is more detailed than STANDARD
        style: dict - the style settings for each verbosity level
        verbosity: int = 0
            the current verbosity level, which is used to control the print output
    """

    STANDARD = 0
    DEBUG = 1

    style = {
        "DEBUG": style.Style(color="dark_orange"),
        "STANDARD": style.Style(color="white"),
    }

    def __init__(self, verbosity: int = STANDARD, **kwargs) -> None:
        self._extra_kwargs = kwargs  # instance-level overrides, merged at print time
        self._verbosity = verbosity

    @property
    def console_kwargs(self) -> dict:
        """Merge global config settings with instance-level overrides at call time."""
        kwargs = get_console_settings()
        kwargs.update(self._extra_kwargs)
        return kwargs

    def writer(self, string: str, *args, verbosity: int = STANDARD, **kwargs) -> None:
        if verbosity >= self._verbosity:
            level_name = "DEBUG" if verbosity >= self.DEBUG else "STANDARD"
            console = Console(**self.console_kwargs)
            console.print(string, *args, **kwargs, style=self.style[level_name])

    def __call__(self, string: str, *args, **kwargs) -> None:
        self.writer(string, *args, verbosity=self.STANDARD, **kwargs)

    def debug(self, string: str, *args, **kwargs) -> None:
        self.writer(string, *args, verbosity=self.DEBUG, **kwargs)

    def print_header(self, string: str) -> None:
        console = Console(**self.console_kwargs)
        console.rule(string)

    def print_table(self, table_column: list, table_row: list, **table_kwargs) -> None:
        console = Console(**self.console_kwargs)

        table = Table(**table_kwargs)
        for column in table_column:
            table.add_column(column)

        for row in table_row:
            table.add_row(*row)

        console.print(table)

    def print_panel(
        self, panel_content: str, panel_title: Optional[str] = None
    ) -> None:
        console = Console(**self.console_kwargs)
        panel = Panel(panel_content, title=panel_title)
        console.print(panel)

    @staticmethod
    def update_settings(**kwargs) -> None:
        update_print_settings(**kwargs)

    @property
    def verbosity(self) -> int:
        if _config.verbosity is None:
            verbosity = self._verbosity
        elif _config.verbosity > self._verbosity:
            verbosity = _config.verbosity
        else:
            verbosity = self._verbosity
        return verbosity
