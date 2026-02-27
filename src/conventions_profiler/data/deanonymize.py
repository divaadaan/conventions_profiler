"""Replace ASAP anonymization tokens with plausible English substitutions."""

import re

PERSON = [
    "James", "Sarah", "David", "Emily", "Michael", "Rachel", "Thomas",
    "Laura", "Robert",
]

LOCATION = [
    "Springfield", "Riverside", "Lakewood", "Fairview", "Greenville",
    "Madison", "Clinton", "Franklin", "Arlington", "Chester",
    "Ashland", "Burlington",
]

ORGANIZATION = [
    "Westfield", "Brookhaven", "Oakridge", "Pinecrest", "Maplewood",
    "Cedarwood",
]

DATE = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

TIME = ["noon", "morning", "evening", "midnight"]

MONEY = ["fifty dollars", "twenty dollars", "ten dollars", "five dollars"]

PERCENT = [
    "thirty percent", "twenty percent", "forty percent", "fifty percent",
    "sixty percent", "ten percent", "seventy percent",
]

NUM = [
    "five", "twelve", "eight", "three", "twenty", "fifteen", "seven",
    "sixteen", "forty", "nine", "eleven", "fourteen", "six",
]

DR = ["Dr.", "Dr."]

CITY = ["Portland"]

STATE = ["Oregon"]

EMAIL = ["name@email.com"]

_POOLS = {
    "PERSON": PERSON,
    "LOCATION": LOCATION,
    "ORGANIZATION": ORGANIZATION,
    "DATE": DATE,
    "TIME": TIME,
    "MONEY": MONEY,
    "PERCENT": PERCENT,
    "NUM": NUM,
    "DR": DR,
    "CITY": CITY,
    "STATE": STATE,
    "EMAIL": EMAIL,
}

# Matches @TYPE followed by digits, e.g. @PERSON1, @LOCATION12
_TOKEN_RE = re.compile(r"@([A-Z]+?)(\d+)")


def deanonymize(text: str) -> str:
    """Replace ASAP anonymization tokens with plausible substitutions."""
    def _replace(match: re.Match) -> str:
        token_type = match.group(1)
        token_index = int(match.group(2))

        if token_type == "MONTH":
            return "May"

        if token_type == "CAPS":
            return match.group(0)

        pool = _POOLS.get(token_type)
        if pool is None:
            return match.group(0)  # unknown leave unchanged

        # 1-based index into pool, wrapping if index exceeds pool size
        return pool[(token_index - 1) % len(pool)]

    return _TOKEN_RE.sub(_replace, text)
