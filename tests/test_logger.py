import pytest
import logging
from trolo.utils.logging.glob_logger import platform_safe_emojis, configure_logger, add_separator_method

@pytest.fixture
def test_logger():
    """Fixture to create a test logger with separator support."""
    logger = configure_logger(name="test_logger", verbose=True)
    return add_separator_method(logger)


def test_platform_safe_emojis():
    """Test that emojis are correctly processed."""
    assert platform_safe_emojis(":fire:") == "🔥"
    assert platform_safe_emojis(":warning:") == "⚠️"
    assert platform_safe_emojis("plain text") == "plain text"


def test_logger_separator(test_logger, capsys):
    """Test the logger's separator method."""
    test_logger.separator("")
    captured = capsys.readouterr().out
    assert "" in captured


def test_logger_configuration():
    """Test that the logger is configured correctly."""
    logger = configure_logger(name="dummy_logger", verbose=True)
    assert logger.name == "dummy_logger"
    assert logger.level == logging.INFO
