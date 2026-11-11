"""Patient grouping helpers for CHB-MIT case-level recordings."""

from __future__ import annotations


# CHB-MIT case chb21 is a later recording from the same person as chb01.
# Keep both cases on the same side of every train/validation/test split.
_CASE_GROUPS = {
    "chb01": "chb01_chb21",
    "chb21": "chb01_chb21",
}


def patient_group(case_id: str) -> str:
    """Return the identity group for a CHB-MIT case (case ID if unique)."""
    normalized = case_id.strip().lower()
    return _CASE_GROUPS.get(normalized, normalized)


def validation_case(case_ids: list[str], test_case: str) -> str:
    """Choose the next case from a person distinct from the held-out test person."""
    test_index = case_ids.index(test_case)
    test_group = patient_group(test_case)
    for offset in range(1, len(case_ids)):
        candidate = case_ids[(test_index + offset) % len(case_ids)]
        if patient_group(candidate) != test_group:
            return candidate
    raise ValueError("LOSO requires a validation patient distinct from the test patient.")


def training_cases(case_ids: list[str], test_case: str, validation: str) -> list[str]:
    """Exclude every case belonging to the test and validation identities."""
    held_out_groups = {patient_group(test_case), patient_group(validation)}
    return [case for case in case_ids if patient_group(case) not in held_out_groups]
