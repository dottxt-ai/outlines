"""Generate valid airport codes."""
from enum import Enum

try:
    from pyairports.airports import AIRPORT_LIST
except ImportError:
    raise ImportError(
        'The `airports` module requires "pyairports" to be installed. You can install it with "pip install pyairports"'
    )


AIRPORT_IATA_LIST = list(
    {(airport[3], airport[3]) for airport in AIRPORT_LIST if airport[3] != ""}
)

IATA = Enum("Airport", AIRPORT_IATA_LIST)  # type:ignore
