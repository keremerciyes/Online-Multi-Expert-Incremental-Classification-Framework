import re


def speed_sort_key(speed: str | int | float) -> tuple[float, str]:
    text = str(speed)
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if numbers:
        return float(numbers[0]), text
    return float("inf"), text


def parse_speed_label(speed_folder: str) -> str:
    return speed_folder.strip()
