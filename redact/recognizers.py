"""Custom Presidio recognizers tailored to this dataset's PHI shapes:
accident-anchored dates, "Lastname, Firstname" style names (classified by
surrounding context into patient/attorney - provider names are left
unredacted, since a treating provider's name identifies the provider, not
the patient, and isn't one of the HIPAA Safe Harbor identifiers, which
only cover identifiers of the individual/patient or their relatives,
employers, or household members), account numbers, and the patient's home
address/phone (identified via the "Home:" label so the clinic's own
letterhead address/phone are left untouched)."""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from presidio_analyzer import EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts

from .date_logic import DATE_PATTERN_REGEXES, DATE_SUBGROUP_REGEXES

# OCR sometimes renders the "Lastname, Firstname" comma as a semicolon.
NAME_RE = re.compile(r"\b[A-Z][A-Za-z'\-]+\s*[,;]\s*[A-Z][a-z]+\b")

# "The Law Office of Zayed Al Sayyed" style attorney mentions. OCR
# sometimes reorders this to "The Office of Zayed Al Sayyed Law", so match
# both "Law"-before and "Law"-after variants.
ATTORNEY_NAME_RE = re.compile(
    r"(?:Law Office of|Attorney)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})"
    r"|Office of\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})\s+Law\b"
)

ATTORNEY_CONTEXT = re.compile(r"(law office|attorney|esq\.?)", re.IGNORECASE)
# A "Lastname, Firstname" match next to a provider-signature label (e.g.
# "Reported by: Agrait-Bertran, Edgardo M.D") must never fall through to
# PATIENT_CONTEXT below - this document is dense enough that such a match
# regularly sits within CONTEXT_WINDOW of an unrelated nearby DOB/Acc No
# header purely by document density, which would otherwise
# misclassify the provider's name as PATIENT_NAME.
PROVIDER_SIGNATURE_CONTEXT = re.compile(
    r"(signed by|reported by|progress notes?:|provider:|\bmd\b|\bm\.d\b|"
    r"\bdpt\b|\bpt,|\bfnp\b|electronically signed)",
    re.IGNORECASE,
)
# Only treat a "Word, Word" match as the patient's name if it actually sits
# near patient-identifying context - this doc is full of unrelated
# comma-separated clinical term pairs ("Ultrasound, Strapping", "Visit,
# Est") that would otherwise default-classify as PATIENT_NAME.
PATIENT_CONTEXT = re.compile(
    r"(DOB\s*:|Acc(?:ount)?\.?\s*No|Account Number|DOS\s*:|Patient\s*Name)", re.IGNORECASE
)
# Billing/procedure-code bullet lines ("- 99213 Office Visit, Est Pt.")
# regularly sit within CONTEXT_WINDOW of a nearby DOB/Acc No header block
# purely by document density, which used to false-positive comma pairs
# like "Visit, Est" or "Exercises, Units" as PATIENT_NAME. A real patient
# name is never itself on a billing-code line, so exclude those outright.
PROCEDURE_CODE_LINE_RE = re.compile(r"-\s*\d{4,5}\b")

CONTEXT_WINDOW = 60


class AccidentDateRecognizer(EntityRecognizer):
    """Detects calendar dates (numeric or month-name) so they can later be
    replaced with an accident-relative offset or a generic [DATE]."""

    ENTITY = "ACCIDENT_DATE"

    def __init__(self):
        super().__init__(supported_entities=[self.ENTITY], name="AccidentDateRecognizer")

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts: Optional[NlpArtifacts]) -> List[RecognizerResult]:
        results = []
        seen_spans = set()

        def _add(start: int, end: int):
            span = (start, end)
            if span in seen_spans:
                return
            seen_spans.add(span)
            results.append(RecognizerResult(entity_type=self.ENTITY, start=start, end=end, score=0.9))

        for pattern in DATE_PATTERN_REGEXES:
            for match in pattern.finditer(text):
                _add(match.start(), match.end())

        for pattern in DATE_SUBGROUP_REGEXES:
            for match in pattern.finditer(text):
                _add(match.start(1), match.end(match.re.groups))

        return results


class NameRecognizer(EntityRecognizer):
    """Detects "Lastname, Firstname" style names and classifies each hit
    as a patient or attorney name based on nearby context. Provider names
    are deliberately not detected/redacted - see module docstring."""

    PATIENT = "PATIENT_NAME"
    ATTORNEY = "ATTORNEY_NAME"

    def __init__(self):
        super().__init__(
            supported_entities=[self.PATIENT, self.ATTORNEY],
            name="NameRecognizer",
        )

    def load(self) -> None:
        pass

    def _classify(self, text: str, start: int, end: int) -> Optional[str]:
        window_start = max(0, start - CONTEXT_WINDOW)
        window_end = min(len(text), end + CONTEXT_WINDOW)
        window = text[window_start:window_end]
        if ATTORNEY_CONTEXT.search(window):
            return self.ATTORNEY
        if PROVIDER_SIGNATURE_CONTEXT.search(window):
            return None
        if PATIENT_CONTEXT.search(window):
            line_start = text.rfind("\n", 0, start) + 1
            if PROCEDURE_CODE_LINE_RE.match(text, line_start):
                return None
            return self.PATIENT
        return None

    def analyze(self, text: str, entities: List[str], nlp_artifacts: Optional[NlpArtifacts]) -> List[RecognizerResult]:
        results = []
        claimed_spans = []

        def _overlaps(start: int, end: int) -> bool:
            return any(start < c_end and end > c_start for c_start, c_end in claimed_spans)

        for match in NAME_RE.finditer(text):
            entity_type = self._classify(text, match.start(), match.end())
            if entity_type is None:
                continue
            results.append(
                RecognizerResult(entity_type=entity_type, start=match.start(), end=match.end(), score=0.85)
            )
            claimed_spans.append((match.start(), match.end()))

        for match in ATTORNEY_NAME_RE.finditer(text):
            group_index = 1 if match.group(1) else 2
            start, end = match.start(group_index), match.end(group_index)
            if _overlaps(start, end):
                continue
            results.append(RecognizerResult(entity_type=self.ATTORNEY, start=start, end=end, score=0.85))
            claimed_spans.append((start, end))

        # Second pass, PATIENT only: OCR/documentation sometimes drops one
        # half of a "Lastname, Firstname" match (e.g. a provider writes the
        # patient's first name directly instead of "Patient"), leaving a
        # bare name fragment elsewhere with no comma and no nearby context.
        # Once we reliably know the patient's last/first name components
        # from clean, context-gated PATIENT matches above, sweep for
        # standalone recurrences of those exact words at a lower score.
        # Scoped to PATIENT only (never ATTORNEY, where common given/family
        # names are far more likely to collide with ordinary words
        # elsewhere in a clinical note).
        known_patient_words = {
            word
            for result in results
            if result.entity_type == self.PATIENT
            for word in re.findall(r"[A-Za-z'\-]+", text[result.start : result.end])
        }
        for word in known_patient_words:
            for match in re.finditer(r"(?<![\w,])" + re.escape(word) + r"(?![\w,])", text):
                start, end = match.start(), match.end()
                if _overlaps(start, end):
                    continue
                results.append(RecognizerResult(entity_type=self.PATIENT, start=start, end=end, score=0.6))
                claimed_spans.append((start, end))

        return results


class AccountNumberRecognizer(EntityRecognizer):
    """Matches 'Acc No 12345' / 'Account Number: 12345', tagging only the
    digits so the field label stays readable."""

    ENTITY = "ACCOUNT_NUMBER"
    PATTERN = re.compile(
        r"acc(?:ount)?\.?\s*(?:no\s*\.?|number)\s*:?\s*(\d{4,10})", re.IGNORECASE
    )

    def __init__(self):
        super().__init__(supported_entities=[self.ENTITY], name="AccountNumberRecognizer")

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts: Optional[NlpArtifacts]) -> List[RecognizerResult]:
        results = []
        claimed_spans = []
        known_numbers = set()

        for match in self.PATTERN.finditer(text):
            start, end = match.start(1), match.end(1)
            results.append(RecognizerResult(entity_type=self.ENTITY, start=start, end=end, score=0.9))
            claimed_spans.append((start, end))
            known_numbers.add(match.group(1))

        # Second pass: the same account number sometimes recurs under a
        # mislabeled/garbled field (e.g. OCR noise placing it after "DOB:").
        # Once we know the real value from a clearly-labeled occurrence,
        # redact every other standalone occurrence of it too.
        for number in known_numbers:
            for match in re.finditer(r"\b" + re.escape(number) + r"\b", text):
                start, end = match.start(), match.end()
                if any(start < c_end and end > c_start for c_start, c_end in claimed_spans):
                    continue
                results.append(RecognizerResult(entity_type=self.ENTITY, start=start, end=end, score=0.75))
                claimed_spans.append((start, end))

        return results


class PatientContactRecognizer(EntityRecognizer):
    """Matches the patient's home address (the block just before a 'Home:'
    label) and home phone number (right after that label). Clinic/provider
    addresses and phone numbers elsewhere in the document are left alone
    since they aren't the patient's own contact info."""

    ADDRESS = "PATIENT_ADDRESS"
    PHONE = "PATIENT_PHONE"

    # "Home:" is sometimes OCR'd as "Homc:" - tolerate any 4th letter.
    ADDRESS_PATTERN = re.compile(
        r"\d{2,6}\s+[NSEW]\.?\s+[\w\s;,\-]{2,40}?\d{5}(?:-\d{4})?(?=\s*Hom[a-z]\s*:)",
        re.IGNORECASE,
    )
    PHONE_PATTERN = re.compile(r"(?<=Hom[a-z]:\s)(\d{3}-\d{3}-\d{4})", re.IGNORECASE)

    def __init__(self):
        super().__init__(
            supported_entities=[self.ADDRESS, self.PHONE], name="PatientContactRecognizer"
        )

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts: Optional[NlpArtifacts]) -> List[RecognizerResult]:
        results = []
        for match in self.ADDRESS_PATTERN.finditer(text):
            results.append(
                RecognizerResult(entity_type=self.ADDRESS, start=match.start(), end=match.end(), score=0.85)
            )
        for match in self.PHONE_PATTERN.finditer(text):
            results.append(
                RecognizerResult(entity_type=self.PHONE, start=match.start(1), end=match.end(1), score=0.85)
            )
        return results


class AgeSexRecognizer(EntityRecognizer):
    """Matches the patient's age+sex, which recurs throughout this dataset
    in a few fixed shapes: "(55 yo F)", "55 Y old Female", "55 y/o female",
    "55 year-old female". Age and sex are tagged as separate entities since
    they're redacted independently."""

    AGE = "AGE"
    SEX = "SEX"

    # Each alternative needs its own group names since Python's re module
    # doesn't allow reusing a group name across "|" branches.
    PATTERN = re.compile(
        r"\((?P<age1>\d{1,3})\s*yo\s*(?P<sex1>[MF])\)"
        r"|(?P<age2>\d{1,3})\s*Y\s*old\s*(?P<sex2>Female|Male)"
        r"|(?P<age3>\d{1,3})\s*y/o\s*(?P<sex3>female|male)"
        r"|(?P<age4>\d{1,3})\s*year-old\s*(?P<sex4>female|male)",
        re.IGNORECASE,
    )

    def __init__(self):
        super().__init__(supported_entities=[self.AGE, self.SEX], name="AgeSexRecognizer")

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts: Optional[NlpArtifacts]) -> List[RecognizerResult]:
        results = []
        for match in self.PATTERN.finditer(text):
            for n in (1, 2, 3, 4):
                age = match.group(f"age{n}")
                if age is None:
                    continue
                results.append(
                    RecognizerResult(
                        entity_type=self.AGE, start=match.start(f"age{n}"), end=match.end(f"age{n}"), score=0.85
                    )
                )
                results.append(
                    RecognizerResult(
                        entity_type=self.SEX, start=match.start(f"sex{n}"), end=match.end(f"sex{n}"), score=0.85
                    )
                )
                break
        return results


class StateRecognizer(EntityRecognizer):
    """Matches a 'State: XX' field (US state abbreviation), e.g. the intake
    form's 'State: NY'. Scoped to this exact labeled-field shape rather than
    a bare 2-letter-code sweep, since a lone uppercase pair is too easy to
    false-positive on elsewhere in clinical text (units, initials, etc.)."""

    ENTITY = "STATE"
    PATTERN = re.compile(r"\bState\s*:\s*([A-Z]{2})\b")

    def __init__(self):
        super().__init__(supported_entities=[self.ENTITY], name="StateRecognizer")

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts: Optional[NlpArtifacts]) -> List[RecognizerResult]:
        return [
            RecognizerResult(entity_type=self.ENTITY, start=match.start(1), end=match.end(1), score=0.85)
            for match in self.PATTERN.finditer(text)
        ]


class PatientDemographicRecognizer(EntityRecognizer):
    """Naive verbatim search-and-replace, driven by an out-of-band patient
    demographic record (see load_patient_demographics) instead of a shape
    pattern. Every non-empty field value is searched for as an exact,
    case-insensitive, whole-word match anywhere in the document - this
    catches recurrences the shape/context recognizers above miss (e.g. a
    preferred name that doesn't follow the "Lastname, Firstname" pattern
    NameRecognizer looks for), but only for values already known ahead of
    time. It can't find PHI it wasn't told about, so it's a supplement to
    the other recognizers here, not a replacement for them."""

    EMAIL = "EMAIL"
    SSN = "SSN"

    # Maps each demographic field to the entity type it's redacted as.
    # Reuses the other recognizers' entity types where one already exists
    # (e.g. a phone number found this way gets the same [PHONE NUMBER]
    # label PatientContactRecognizer produces) so downstream labeling stays
    # consistent regardless of which recognizer caught a given hit.
    FIELD_ENTITY_MAP = {
        "Preferred Name": NameRecognizer.PATIENT,
        "Previous Name": NameRecognizer.PATIENT,
        "Prefix": NameRecognizer.PATIENT,
        "Suffix": NameRecognizer.PATIENT,
        "Sex": AgeSexRecognizer.SEX,
        "Cell Phone": PatientContactRecognizer.PHONE,
        "Home Phone": PatientContactRecognizer.PHONE,
        "Work Phone": PatientContactRecognizer.PHONE,
        "Email": EMAIL,
        "SSN": SSN,
        "Acc No": AccountNumberRecognizer.ENTITY,
    }

    # A naive whole-value match shorter than this is too likely to collide
    # with an ordinary word/abbreviation elsewhere in the document (e.g. a
    # single-letter "Sex": "M" would redact every standalone "M" in the
    # text) - skip it rather than over-redact.
    MIN_VALUE_LENGTH = 3

    def __init__(self, demographics: Dict[str, str]):
        super().__init__(
            supported_entities=sorted(set(self.FIELD_ENTITY_MAP.values())),
            name="PatientDemographicRecognizer",
        )
        self.demographics = demographics

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts: Optional[NlpArtifacts]) -> List[RecognizerResult]:
        results = []
        for field, entity_type in self.FIELD_ENTITY_MAP.items():
            value = (self.demographics.get(field) or "").strip()
            if len(value) < self.MIN_VALUE_LENGTH:
                continue
            # Not \b: a value ending in punctuation (e.g. "Prefix": "Ms.")
            # has no word/non-word *transition* right after it when
            # followed by a space, so \b would silently never match it.
            # (?<!\w)/(?!\w) check the adjacent character in isolation
            # instead, which handles that correctly.
            pattern = re.compile(
                r"(?<!\w)" + re.escape(value) + r"(?!\w)", re.IGNORECASE
            )
            for match in pattern.finditer(text):
                results.append(
                    RecognizerResult(entity_type=entity_type, start=match.start(), end=match.end(), score=1.0)
                )
        return results


def load_patient_demographics(path: Path) -> Dict[str, str]:
    """Load a single patient's demographic record from a JSON file (see
    data/patient_demographics.json for the expected fields). Missing or
    empty fields are simply skipped by PatientDemographicRecognizer, so a
    partially-filled record is fine."""
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def build_custom_recognizers(patient_demographics: Optional[Dict[str, str]] = None):
    recognizers = [
        AccidentDateRecognizer(),
        NameRecognizer(),
        AccountNumberRecognizer(),
        PatientContactRecognizer(),
        AgeSexRecognizer(),
        StateRecognizer(),
    ]
    if patient_demographics:
        recognizers.append(PatientDemographicRecognizer(patient_demographics))
    return recognizers
