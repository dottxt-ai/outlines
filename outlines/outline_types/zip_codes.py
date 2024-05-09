"""Zip code types.

We currently only support US Zip Codes.

"""
from pydantic import WithJsonSchema
from typing_extensions import Annotated

# This matches Zip and Zip+4 codes
US_ZIP_CODE = r"\d{5}(?:-\d{4})?"


USZipCode = Annotated[str, WithJsonSchema({"type": "string", "pattern": US_ZIP_CODE})]
