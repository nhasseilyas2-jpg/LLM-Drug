"""Python helpers for the LLM Injection Runtime Lab.

Stdlib only. ``to_dataframe`` needs pandas, which is optional.
"""

from .client import LabClient, LabError
from .records import COLUMNS, flatten, load_history, to_dataframe

__all__ = ["LabClient", "LabError", "COLUMNS", "flatten", "load_history", "to_dataframe"]
__version__ = "0.1.0"
