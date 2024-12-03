from colorama import init, Fore, Style
import emoji
import sys
import logging

def platform_safe_emojis(emoji_str=""):
    """Return emoji-safe version of the string."""
    return emoji.emojize(emoji_str, language='alias')

class ColorLogger(logging.Formatter):
    """
    Custom formatter with colors & emojis for all log levels.
    """
    # No color for INFO
    COLORS = {
        "DEBUG": Fore.BLUE,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.MAGENTA,
    }

    EMOJIS = {
        "DEBUG": "🐛",
        "INFO": "🛈",  # Info symbol
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "🔥",
    }
    def format(self, record):
        levelname = record.levelname
        message = super().format(record)
        color = self.COLORS.get(levelname, Style.RESET_ALL)
        emoji_symbol = platform_safe_emojis(self.EMOJIS.get(levelname, ""))
        return f"{color}{emoji_symbol} {message}{Style.RESET_ALL}"

def add_separator_method(logger):
    """
    Add a separator method to the logger.
    
    :param logger: The logger instance
    :return: The logger with added separator method
    """
    def separator(text, char='-', width=85):
        """
        Create a separator line with optional text.
        
        :param text: Text to display in the separator
        :param char: Character to use for separator
        :param width: Total width of the separator
        """
        # Calculate padding
        total_width = width
        text_with_spaces = f" {text} "
        padding_length = (total_width - len(text_with_spaces)) // 2
        
        # Create separator line
        separator_line = char * padding_length + text_with_spaces + char * padding_length
        
        # Ensure exact width by trimming or padding
        separator_line = separator_line[:total_width]
        separator_line = separator_line.ljust(total_width, char)
        
        # Log the separator
        logger.info(separator_line)
    
    # Attach the separator method to the logger
    logger.separator = separator
    return logger

def configure_logger(name="default_logger", verbose=True, rank=0):
    """
    Configure logger with color and emoji formatting and customizable verbosity.
    
    :param name: Name of the logger
    :param verbose: Whether to set logging level to INFO
    :param rank: Rank of the process (to control logging)
    """
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    formatter = ColorLogger("%(message)s")

    # StreamHandler setup
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    # Configure the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False

    return logger

# Lazy initialization
LOGGER =  add_separator_method(configure_logger())