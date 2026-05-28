"""The utility module for defining some simple utility functions, which is directly used in IO, and so on."""

from rich import style
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Default print settings

class Printer:
    STANDARD = 0
    DEBUG = 1

    style = {
        'DEBUG': style.Style(color='dark_orange'),
        'STANDARD': style.Style(color='white'),
    }

    def __init__(self, verbosity: int = STANDARD, **kwargs) -> None:
        self.console_kwargs = get_printer_settings(verbosity)
        self.console_kwargs.update(kwargs)
        self._verbosity = verbosity

    def writer(self, string: str, *args, verbosity: int = STANDARD, **kwargs) -> None:
        if verbosity >= self._verbosity:
            console = Console(**self.console_kwargs)
            console.print(string, *args, **kwargs, style=self.style[verbosity])

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

    def print_panel(self, panel_content: str, panel_title: Optional[str] = None) -> None:
        console = Console(**self.console_kwargs)
        panel = Panel(panel_content, title=panel_title)
        console.print(panel)

    @staticmethod
    def update_settings(**kwargs) -> None:
        update_print_settings(**kwargs)

    @property
    def verbosity(self) -> int:
        
