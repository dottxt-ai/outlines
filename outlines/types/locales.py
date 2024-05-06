from dataclasses import dataclass

from outlines.types.phone_numbers import USPhoneNumber
from outlines.types.zip_codes import USZipCode


@dataclass
class US:
    ZipCode = USZipCode
    PhoneNumber = USPhoneNumber


def locale(locale_str: str):
    locales = {"us": US}

    if locale_str not in locales:
        raise NotImplementedError(
            f"The locale {locale_str} is not supported yet. Please don't hesitate to create custom types for you locale and open a Pull Request."
        )

    return locales[locale_str]
