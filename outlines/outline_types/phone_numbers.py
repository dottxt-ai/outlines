"""Phone number types.

We currently only support US phone numbers. We can however imagine having custom types
for each country, for instance leveraging the `phonenumbers` library.

"""
from pydantic import WithJsonSchema
from typing_extensions import Annotated

US_PHONE_NUMBER = r"(\([0-9]{3}\) |[0-9]{3}-)[0-9]{3}-[0-9]{4}"


USPhoneNumber = Annotated[
    str,
    WithJsonSchema({"type": "string", "pattern": US_PHONE_NUMBER}),
]
