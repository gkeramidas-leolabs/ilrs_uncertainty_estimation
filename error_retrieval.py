from typing import List

ILRS = {}
ILRS["L5011"] = [0, 0, 0, 0, 0, 0]
ILRS["L3059"] = [0, 0, 0, 0, 0, 0]
ILRS["L335"] = [0, 0, 0, 0, 0, 0]
ILRS["L2486"] = [0, 0, 0, 0, 0, 0]
ILRS["L4884"] = [0, 0, 0, 0, 0, 0]
ILRS["L1471"] = [0, 0, 0, 0, 0, 0]
ILRS["L5429"] = [0, 0, 0, 0, 0, 0]
ILRS["L3972"] = [0, 0, 0, 0, 0, 0]
ILRS["L3969"] = [0, 0, 0, 0, 0, 0]
ILRS["L2669"] = [0, 0, 0, 0, 0, 0]
ILRS["L2682"] = [0, 0, 0, 0, 0, 0]
ILRS["L3226"] = [0, 0, 0, 0, 0, 0]


def get_ilrs_uncertainty(leolabs_id: str) -> List:
    return ILRS[str(leolabs_id)]
