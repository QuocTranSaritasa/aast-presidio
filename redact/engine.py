"""Wires the custom recognizers into a Presidio AnalyzerEngine and builds
the matching AnonymizerEngine operator config (bracketed labels, with a
custom lambda for the accident-relative date offsets).

Also supports two evaluation modes that additionally run a generic NER
model side by side with our custom recognizers, so gaps in the custom
rules show up distinctly - anything only the NER model caught is rendered
with double brackets, e.g. [[PERSON]], instead of our normal [PATIENT NAME]
style single-bracket labels:

  - "spacy": Presidio's stock SpacyRecognizer over spaCy's generic NER
    (PERSON/LOCATION). Runs in the normal .venv.
  - "clinical": the same mechanism, but backed by a transformer model
    fine-tuned on the i2b2 clinical de-identification dataset
    (obi/deid_roberta_i2b2) instead of generic spaCy NER. This needs
    torch/transformers/spacy-huggingface-pipelines, which conflict with
    this project's opencv/numpy versions on this platform - see
    requirements-clinical.txt and run it from the separate .venv-clinical
    (via redact_clinical_eval.py), never the main .venv. The heavy imports
    are therefore deferred to build_clinical_analyzer_engine() so just
    importing this module never requires torch to be installed."""
from typing import List, Optional, Tuple

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import SpacyRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from .date_logic import find_accident_anchor, parse_date_text, redact_date_text
from .recognizers import (
    AccidentDateRecognizer,
    AccountNumberRecognizer,
    AgeSexRecognizer,
    NameRecognizer,
    PatientContactRecognizer,
    StateRecognizer,
    build_custom_recognizers,
)

CUSTOM_ENTITIES = [
    NameRecognizer.PATIENT,
    NameRecognizer.PROVIDER,
    NameRecognizer.ATTORNEY,
    AccidentDateRecognizer.ENTITY,
    AccountNumberRecognizer.ENTITY,
    PatientContactRecognizer.ADDRESS,
    PatientContactRecognizer.PHONE,
    AgeSexRecognizer.AGE,
    AgeSexRecognizer.SEX,
    StateRecognizer.ENTITY,
]

# Presidio's SpacyRecognizer, used only for side-by-side evaluation against
# our custom recognizers - never mixed into the "real" redaction output.
#
# spaCy's NER has no real per-entity confidence: every hit gets the same
# flat 0.85 score (see NerModelConfiguration.default_score), so a score
# threshold can't separate "confident" from "iffy" matches here - the only
# effective lever is which entity types we even ask for. ORGANIZATION and
# DATE_TIME were almost pure noise on this OCR'd clinical text (ICD codes,
# abbreviations like "PT"/"SVINT", and vague words like "today" all got
# tagged), so they're dropped. PERSON and LOCATION were where spaCy
# actually surfaced real, useful gaps (e.g. catching the accident city
# "Newyork" that our custom recognizers don't cover).
SPACY_ENTITIES = ["PERSON", "LOCATION"]

# obi/deid_roberta_i2b2 is fine-tuned directly on the i2b2 medical
# de-identification corpus, and Presidio ships a matching default label
# mapping for it (PATIENT/STAFF -> PERSON, HOSP/PATORG -> ORGANIZATION,
# DATE -> DATE_TIME, PHONE -> PHONE_NUMBER, plus AGE/ID/EMAIL/LOCATION
# unchanged) - see presidio_analyzer.nlp_engine.ner_model_configuration.
# Unlike spaCy, this model produces real per-entity confidence scores
# (softmax probabilities), not a flat constant, so a score threshold here
# is actually meaningful.
CLINICAL_ENTITIES = [
    "PERSON",
    "LOCATION",
    "ORGANIZATION",
    "DATE_TIME",
    "AGE",
    "ID",
    "EMAIL",
    "PHONE_NUMBER",
    "NRP",
]

LABELS = {
    NameRecognizer.PATIENT: "[PATIENT NAME]",
    NameRecognizer.PROVIDER: "[PROVIDER NAME]",
    NameRecognizer.ATTORNEY: "[ATTORNEY NAME]",
    AccountNumberRecognizer.ENTITY: "[ACCOUNT NUMBER]",
    PatientContactRecognizer.ADDRESS: "[ADDRESS]",
    PatientContactRecognizer.PHONE: "[PHONE NUMBER]",
    AgeSexRecognizer.AGE: "[AGE]",
    AgeSexRecognizer.SEX: "[SEX]",
    StateRecognizer.ENTITY: "[STATE]",
}


def build_analyzer_engine() -> AnalyzerEngine:
    """A spaCy NLP engine, our custom recognizers, and (registered but
    dormant) Presidio's stock SpacyRecognizer. Requesting exactly
    CUSTOM_ENTITIES from analyze() means the registry never invokes
    SpacyRecognizer (its supported entities aren't in that list) - it only
    runs when analyze() is explicitly called with entities=SPACY_ENTITIES,
    i.e. via analyze_with_spacy_eval()."""
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }
    )
    nlp_engine = provider.create_engine()

    registry = RecognizerRegistry()
    for recognizer in build_custom_recognizers():
        registry.add_recognizer(recognizer)
    registry.add_recognizer(SpacyRecognizer(supported_entities=SPACY_ENTITIES))

    return AnalyzerEngine(registry=registry, nlp_engine=nlp_engine, supported_languages=["en"])


def build_clinical_analyzer_engine() -> AnalyzerEngine:
    """Same custom recognizers as build_analyzer_engine(), but backed by
    obi/deid_roberta_i2b2 (a RoBERTa model fine-tuned on the i2b2 clinical
    de-identification dataset) instead of generic spaCy NER, for
    evaluation. Only usable from .venv-clinical - see the module
    docstring for why."""
    from presidio_analyzer.nlp_engine import TransformersNlpEngine

    nlp_engine = TransformersNlpEngine()  # defaults to obi/deid_roberta_i2b2 + en_core_web_sm

    registry = RecognizerRegistry()
    for recognizer in build_custom_recognizers():
        registry.add_recognizer(recognizer)
    registry.add_recognizer(SpacyRecognizer(supported_entities=CLINICAL_ENTITIES))

    return AnalyzerEngine(registry=registry, nlp_engine=nlp_engine, supported_languages=["en"])


# Prefix used to namespace eval-only NER results before anonymizing, so
# they never collide with a same-named production entity type (e.g. our
# custom AgeSexRecognizer and CLINICAL_ENTITIES both use "AGE" - without
# this, AnonymizerEngine's single per-entity-type operator would render
# *every* AGE hit - including the ones our own recognizer correctly
# caught - with the eval's double-bracket label instead of the normal one).
EVAL_ONLY_PREFIX = "EVAL_ONLY__"


def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and a_end > b_start


def analyze_with_eval(
    text: str,
    analyzer: AnalyzerEngine,
    eval_entities: List[str],
    score_threshold: Optional[float] = None,
) -> Tuple[List[RecognizerResult], List[RecognizerResult]]:
    """Run our custom recognizers and the given NER entity set as two
    separate passes (sharing one NLP parse), then keep only the NER hits
    that don't overlap anything our custom rules already found. Returns
    (custom_results, ner_only_results).

    score_threshold only filters the NER pass, not our custom recognizers
    (which always score 0.85-0.9, so a threshold below that wouldn't
    change anything for them anyway). Only meaningful for models with real
    per-entity confidence (e.g. the clinical transformer) - spaCy's NER
    gives every hit the same flat 0.85, so no threshold can separate
    "confident" from "iffy" there."""
    language = "en"
    nlp_artifacts = analyzer.nlp_engine.process_text(text, language)

    custom_results = analyzer.analyze(
        text=text, language=language, entities=CUSTOM_ENTITIES, nlp_artifacts=nlp_artifacts
    )
    ner_results = analyzer.analyze(
        text=text,
        language=language,
        entities=eval_entities,
        nlp_artifacts=nlp_artifacts,
        score_threshold=score_threshold,
    )

    ner_only_results = [
        r
        for r in ner_results
        if not any(_spans_overlap(r.start, r.end, c.start, c.end) for c in custom_results)
    ]
    return custom_results, ner_only_results


def analyze_with_spacy_eval(
    text: str, analyzer: AnalyzerEngine
) -> Tuple[List[RecognizerResult], List[RecognizerResult]]:
    return analyze_with_eval(text, analyzer, SPACY_ENTITIES)


# Below this, obi/deid_roberta_i2b2's own confidence drops off a cliff on
# this dataset: real gaps (missed provider names, citation authors, the
# accident street name) score >=0.617, while outright mistakes ("Thoracic"
# as PERSON) score <=0.575. This threshold sits in that gap.
CLINICAL_SCORE_THRESHOLD = 0.6


def analyze_with_clinical_eval(
    text: str, analyzer: AnalyzerEngine, score_threshold: Optional[float] = CLINICAL_SCORE_THRESHOLD
) -> Tuple[List[RecognizerResult], List[RecognizerResult]]:
    return analyze_with_eval(text, analyzer, CLINICAL_ENTITIES, score_threshold=score_threshold)


def build_anonymizer_operators(anchor, eval_entities: Optional[List[str]] = None):
    def date_lambda(original_text: str) -> str:
        return redact_date_text(original_text, anchor)

    operators = {
        entity_type: OperatorConfig("replace", {"new_value": label})
        for entity_type, label in LABELS.items()
    }
    operators[AccidentDateRecognizer.ENTITY] = OperatorConfig("custom", {"lambda": date_lambda})

    for entity_type in eval_entities or []:
        operators[f"{EVAL_ONLY_PREFIX}{entity_type}"] = OperatorConfig(
            "replace", {"new_value": f"[[{entity_type}]]"}
        )

    return operators


def _namespace_eval_only(results: List[RecognizerResult]) -> List[RecognizerResult]:
    """Rename each result's entity_type with EVAL_ONLY_PREFIX so it gets a
    distinct anonymizer operator even if the raw name (e.g. "AGE") matches
    a production entity type."""
    return [
        RecognizerResult(
            entity_type=f"{EVAL_ONLY_PREFIX}{r.entity_type}",
            start=r.start,
            end=r.end,
            score=r.score,
        )
        for r in results
    ]


def redact_text(
    text: str,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
    accident_date: Optional[str] = None,
    eval_entities: Optional[List[str]] = None,
    eval_score_threshold: Optional[float] = None,
) -> str:
    # If the caller supplies the accident date up front (e.g. it's missing
    # from this particular document, so "Date of Accident: ..." can't be
    # found in the text), use it as the anchor directly. Otherwise fall
    # back to detecting it from the document as before.
    anchor = parse_date_text(accident_date) if accident_date else find_accident_anchor(text)

    if eval_entities:
        custom_results, ner_only_results = analyze_with_eval(
            text, analyzer, eval_entities, score_threshold=eval_score_threshold
        )
        results = custom_results + _namespace_eval_only(ner_only_results)
    else:
        results = analyzer.analyze(text=text, language="en", entities=CUSTOM_ENTITIES)

    operators = build_anonymizer_operators(anchor, eval_entities=eval_entities)
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results, operators=operators)
    return anonymized.text
