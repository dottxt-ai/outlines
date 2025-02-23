"""Generate valid airport codes."""

from enum import Enum

import airportsdata

AIRPORT_IATA_LIST = [
    (v["iata"], v["iata"]) for v in airportsdata.load().values() if v["iata"]
]
IATA = Enum("Airport", AIRPORT_IATA_LIST)  # type:ignore
