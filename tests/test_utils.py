"""Tests for gs_gmlip.utils."""

from __future__ import annotations

import pytest
from io import StringIO
from unittest.mock import patch

from gs_gmlip.utils import (
    DEFAULT_PRINT_SETTINGS,
    PrintConfig,
    Printer,
    get_config,
    get_console_settings,
    update_print_settings,
)


# ---------------------------------------------------------------------------
# PrintConfig
# ---------------------------------------------------------------------------


class TestPrintConfig:
    def test_defaults(self):
        cfg = PrintConfig()
        assert cfg.verbosity == 0
        assert cfg.settings == DEFAULT_PRINT_SETTINGS

    def test_update(self):
        cfg = PrintConfig()
        cfg.update(width=80)
        assert cfg.settings["width"] == 80

    def test_reset(self):
        cfg = PrintConfig()
        cfg.update(width=80)
        cfg.reset()
        assert cfg.settings == DEFAULT_PRINT_SETTINGS

    def test_settings_are_independent_copies(self):
        cfg1 = PrintConfig()
        cfg2 = PrintConfig()
        cfg1.update(width=60)
        assert cfg2.settings["width"] == DEFAULT_PRINT_SETTINGS["width"]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestModuleHelpers:
    def test_get_config_returns_singleton(self):
        assert get_config() is get_config()

    def test_get_console_settings_returns_dict(self):
        settings = get_console_settings()
        assert isinstance(settings, dict)
        assert "width" in settings

    def test_get_console_settings_is_a_copy(self):
        s1 = get_console_settings()
        s2 = get_console_settings()
        s1["width"] = 999
        assert s2["width"] != 999

    def test_update_print_settings_affects_get_console_settings(self):
        original = get_console_settings()["width"]
        update_print_settings(width=60)
        assert get_console_settings()["width"] == 60
        update_print_settings(width=original)  # restore

    def test_update_print_settings_adds_new_key(self):
        update_print_settings(_test_key="hello")
        assert get_console_settings()["_test_key"] == "hello"
        # clean up
        get_config().settings.pop("_test_key", None)


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------


class TestPrinter:
    def test_instantiation_default(self):
        p = Printer()
        assert p._verbosity == Printer.STANDARD

    def test_instantiation_debug(self):
        p = Printer(verbosity=Printer.DEBUG)
        assert p._verbosity == Printer.DEBUG

    def test_verbosity_property_uses_instance_value_when_config_lower(self):
        p = Printer(verbosity=Printer.DEBUG)
        get_config().verbosity = Printer.STANDARD
        assert p.verbosity == Printer.DEBUG

    def test_verbosity_property_uses_config_when_higher(self):
        p = Printer(verbosity=Printer.STANDARD)
        get_config().verbosity = Printer.DEBUG
        assert p.verbosity == Printer.DEBUG
        get_config().verbosity = Printer.STANDARD  # restore

    def test_style_keys_exist(self):
        assert "STANDARD" in Printer.style
        assert "DEBUG" in Printer.style

    def test_writer_prints_at_standard(self, capsys):
        p = Printer(verbosity=Printer.STANDARD)
        with patch("gs_gmlip.utils.Console") as MockConsole:
            instance = MockConsole.return_value
            p.writer("hello", verbosity=Printer.STANDARD)
            instance.print.assert_called_once()

    def test_writer_suppressed_below_verbosity(self):
        p = Printer(verbosity=Printer.DEBUG)
        with patch("gs_gmlip.utils.Console") as MockConsole:
            instance = MockConsole.return_value
            p.writer("hidden", verbosity=Printer.STANDARD)
            instance.print.assert_not_called()

    def test_call_delegates_to_writer_at_standard(self):
        p = Printer(verbosity=Printer.STANDARD)
        with patch.object(p, "writer") as mock_writer:
            p("msg")
            mock_writer.assert_called_once_with("msg", verbosity=Printer.STANDARD)

    def test_debug_delegates_to_writer_at_debug(self):
        p = Printer(verbosity=Printer.STANDARD)
        with patch.object(p, "writer") as mock_writer:
            p.debug("dbg")
            mock_writer.assert_called_once_with("dbg", verbosity=Printer.DEBUG)

    def test_print_header_calls_rule(self):
        p = Printer()
        with patch("gs_gmlip.utils.Console") as MockConsole:
            instance = MockConsole.return_value
            p.print_header("Section")
            instance.rule.assert_called_once_with("Section")

    def test_print_table_renders(self):
        p = Printer()
        with patch("gs_gmlip.utils.Console") as MockConsole:
            instance = MockConsole.return_value
            p.print_table(["A", "B"], [("1", "2"), ("3", "4")])
            instance.print.assert_called_once()

    def test_print_panel_renders(self):
        p = Printer()
        with patch("gs_gmlip.utils.Console") as MockConsole:
            instance = MockConsole.return_value
            p.print_panel("content", panel_title="Title")
            instance.print.assert_called_once()

    def test_update_settings_static(self):
        original = get_console_settings()["width"]
        Printer.update_settings(width=50)
        assert get_console_settings()["width"] == 50
        update_print_settings(width=original)  # restore

    def test_console_kwargs_reflects_global_config_changes(self):
        # Printer created before the config change should still see the new value
        p = Printer()
        original = get_console_settings()["width"]
        update_print_settings(width=77)
        assert p.console_kwargs["width"] == 77
        update_print_settings(width=original)  # restore

    def test_instance_extra_kwargs_override_global(self):
        p = Printer(width=999)
        assert p.console_kwargs["width"] == 999
        # global change must NOT override the instance-level value
        update_print_settings(width=50)
        assert p.console_kwargs["width"] == 999
        update_print_settings(width=DEFAULT_PRINT_SETTINGS["width"])  # restore
