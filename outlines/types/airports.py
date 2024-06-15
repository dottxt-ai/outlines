"""Generate valid airport codes."""
from enum import Enum

from pyairports.airports import AIRPORT_LIST

AIRPORT_IATA_LIST = list(
    {(airport[3], airport[3]) for airport in AIRPORT_LIST if airport[3] != ""}
)

IATA = Enum("Airport", AIRPORT_IATA_LIST)  # type:ignore
