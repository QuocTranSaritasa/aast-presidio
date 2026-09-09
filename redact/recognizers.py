"""Custom Presidio recognizers tailored to this dataset's PHI shapes:
accident-anchored dates, "Lastname, Firstname" style names (classified by
surrounding context into patient/provider/attorney), account numbers, and
the patient's home address/phone (identified via the "Home:" label so the
clinic's own letterhead address/phone are left untouched)."""
import re
from typing import List, Optional

from presidio_analyzer import EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts

from .date_logic import DATE_PATTERN_REGEXES, DATE_SUBGROUP_REGEXES

# OCR sometimes renders the "Lastname, Firstname" comma as a semicolon.
NAME_RE = re.compile(r"\b[A-Z][A-Za-z'\-]+\s*[,;]\s*[A-Z][a-z]+\b")

# "Firstname Lastname" directly followed by a credential, e.g. "Sabrina
# Browning MD" or "Alexander Barrera, DPT" (allow an optional comma before
# the credential for the latter form).
PROVIDER_CREDENTIAL_RE = re.compile(
    r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)(?=,?\s+(?:MD\b|M\.D\.?\b|DPT\b|PT,|FNP))"
)

# Plain "Firstname Lastname" (no comma, no trailing credential) directly
# after a strong provider-identifying label.
LABEL_PROVIDER_RE = re.compile(
    r"(?:Progress Notes|[Ee]lectronically signed by(?:\s+Provider)?)\s*:?\s*"
    r"([A-Z][a-z]+(?:[-\s][A-Z][a-z]+){1,3})"
)

# "The Law Office of Zayed Al Sayyed" style attorney mentions. OCR
# sometimes reorders this to "The Office of Zayed Al Sayyed Law", so match
# both "Law"-before and "Law"-after variants.
ATTORNEY_NAME_RE = re.compile(
    r"(?:Law Office of|Attorney)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})"
    r"|Office of\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})\s+Law\b"
)

PROVIDER_CONTEXT = re.compile(
    r"(signed by|reported by|progress notes?:|provider:|\bmd\b|\bm\.d\b|"
    r"\bdpt\b|\bpt,|\bfnp\b|electronically signed)",
    re.IGNORECASE,
)
ATTORNEY_CONTEXT = re.compile(r"(law office|attorney|esq\.?)", re.IGNORECASE)
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
    as a patient, provider, or attorney name based on nearby context."""

    PATIENT = "PATIENT_NAME"
    PROVIDER = "PROVIDER_NAME"
    ATTORNEY = "ATTORNEY_NAME"

    def __init__(self):
        super().__init__(
            supported_entities=[self.PATIENT, self.PROVIDER, self.ATTORNEY],
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
        if PROVIDER_CONTEXT.search(window):
            return self.PROVIDER
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

        for match in PROVIDER_CREDENTIAL_RE.finditer(text):
            start, end = match.start(1), match.end(1)
            if _overlaps(start, end):
                continue
            results.append(RecognizerResult(entity_type=self.PROVIDER, start=start, end=end, score=0.85))
            claimed_spans.append((start, end))

        for match in LABEL_PROVIDER_RE.finditer(text):
            start, end = match.start(1), match.end(1)
            if _overlaps(start, end):
                continue
            results.append(RecognizerResult(entity_type=self.PROVIDER, start=start, end=end, score=0.85))
            claimed_spans.append((start, end))

        # Second pass, PATIENT only: OCR/documentation sometimes drops one
        # half of a "Lastname, Firstname" match (e.g. a provider writes the
        # patient's first name directly instead of "Patient"), leaving a
        # bare name fragment elsewhere with no comma and no nearby context.
        # Once we reliably know the patient's last/first name components
        # from clean, context-gated PATIENT matches above, sweep for
        # standalone recurrences of those exact words at a lower score.
        # Scoped to PATIENT only (never PROVIDER/ATTORNEY, where common
        # given/family names are far more likely to collide with ordinary
        # words elsewhere in a clinical note).
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


def build_custom_recognizers():
    return [
        AccidentDateRecognizer(),
        NameRecognizer(),
        AccountNumberRecognizer(),
        PatientContactRecognizer(),
        AgeSexRecognizer(),
        StateRecognizer(),
    ]
