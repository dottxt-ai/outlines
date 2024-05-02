"""ISBN type"""
from pydantic import WithJsonSchema
from typing_extensions import Annotated

# Matches any ISBN number.  Note that this is not completely correct as not all
# 10 or 13 digits numbers are valid ISBNs. See https://en.wikipedia.org/wiki/ISBN
# Taken from O'Reilly's Regular Expression Cookbook:
# https://www.oreilly.com/library/view/regular-expressions-cookbook/9781449327453/ch04s13.html
# TODO: Can this be represented by a grammar or do we need semantic checks?
ISBN_REGEX = r"(?:ISBN(?:-1[03])?:? )?(?=[0-9X]{10}$|(?=(?:[0-9]+[- ]){3})[- 0-9X]{13}$|97[89][0-9]{10}$|(?=(?:[0-9]+[- ]){4})[- 0-9]{17}$)(?:97[89][- ]?)?[0-9]{1,5}[- ]?[0-9]+[- ]?[0-9]+[- ]?[0-9X]"

ISBN = Annotated[str, WithJsonSchema({"type": "string", "pattern": ISBN_REGEX})]
