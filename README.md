# presidio-test

PHI redaction for medical records (text + scanned image) built on
[Microsoft Presidio](https://microsoft.github.io/presidio/), using custom
recognizers tuned to this dataset's field shapes rather than relying on
generic NER.

## What it does

- **Text** (`data/encounters.md`): finds patient/attorney names, account
  numbers, home address/phone, age/sex, state, and dates, and replaces
  each with a bracketed label, e.g. `[PATIENT NAME]`, `[ACCOUNT NUMBER]`,
  `[AGE]`.
- **Image** (`data/Patient_Intake.jpg`): same detection, run over OCR'd
  text (Tesseract), drawn back onto the image as black boxes with the
  label baked in.
- **Dates** are anchored to the document's own "Date of Accident": any
  date within 2 years of it is replaced with a relative offset
  (`[113 DAYS POST ACCIDENT]` / `[40 DAYS PRE ACCIDENT]`); anything
  farther out (e.g. a date of birth) becomes a generic `[DATE]`.

Scope note: treating provider names, the clinic's own letterhead
(address/phone/name), and the accident scene's street location are
intentionally left unredacted. None of these are patient identifiers, and
none are on the HIPAA Safe Harbor list of identifiers to remove (45 CFR
164.514(b)(2)(i)), which only covers identifiers of the individual (the
patient) or their relatives, employers, or household members.

## Repo layout

```
redact/
  date_logic.py          accident-date anchoring + offset formatting
  recognizers.py         custom Presidio EntityRecognizers (name, date,
                          account number, address/phone, age/sex, state)
  engine.py               wires recognizers into an AnalyzerEngine +
                          AnonymizerEngine; also the NER-eval machinery
  text_redact.py          redact encounters.md -> encounters.redacted.md
  image_redact.py         redact Patient_Intake.jpg (via
                          presidio-image-redactor, needs opencv)
  clinical_image_redact.py  image redaction for the clinical-eval path
                          (pytesseract directly, no opencv - see below)
  image_draw.py           shared PIL box/label drawing helpers
  ocr_utils.py            minimal OCR + text/bbox mapping (no opencv)
redact.py                 CLI: production redaction + spaCy-eval mode
redact_clinical_eval.py   CLI: clinical-transformer-eval mode (separate venv)
requirements.txt          deps for redact.py (main venv)
requirements-clinical.txt deps for redact_clinical_eval.py (separate venv)
data/                     input files + redacted/ outputs (gitignored)
```

## Setup

### Prerequisites

- Python 3.11
- Tesseract OCR binary on `PATH` (e.g. `brew install tesseract` on macOS)

### Main environment (`.venv`)

Everything except the clinical-transformer eval runs here.

Run this **once** to create the venv and install into it:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

Every time after that, the venv already has everything installed - just
activate it, no need to re-run `pip install`:

```bash
source .venv/bin/activate
```

### Clinical-eval environment (`.venv-clinical`) - optional

Only needed if you want to run `redact_clinical_eval.py` (evaluates the
custom recognizers against `obi/deid_roberta_i2b2`, a transformer
fine-tuned on the i2b2 clinical de-identification dataset).

This is a **separate venv on purpose**: on this platform, the newest
available `torch` build (macOS x86_64 wheels stop at 2.2.2) needs
`numpy<2`, while `presidio-image-redactor` (used by the main venv) pins
`opencv-python>=4.13`, which itself requires `numpy>=2`. Those can't both
be satisfied in one environment, so the clinical-eval path avoids
presidio-image-redactor entirely and talks to Tesseract directly instead
(see `redact/ocr_utils.py`).

Run this **once** to create the venv and install into it (this downloads
`torch`, so it's a bigger/slower install than the main venv):

```bash
python3.11 -m venv .venv-clinical
source .venv-clinical/bin/activate
pip install -r requirements-clinical.txt
python -m spacy download en_core_web_sm
```

Every time after that, just activate - no need to `pip install` again:

```bash
source .venv-clinical/bin/activate
```

(Or skip activating altogether and call the venv's interpreter directly,
e.g. `.venv-clinical/bin/python redact_clinical_eval.py`.)

## Usage

All commands below assume the relevant venv is already activated (see
Setup above - `pip install` is a one-time step, not needed before every
run).

### Production redaction (`.venv`)

```bash
python redact.py                 # redact both data/encounters.md and data/Patient_Intake.jpg
python redact.py --text-only     # just the text file
python redact.py --image-only    # just the image
```

Output goes to `data/redacted/encounters.redacted.md` and
`data/redacted/Patient_Intake.redacted.jpg`.

### spaCy NER evaluation (`.venv`)

Runs the production recognizers side by side with Presidio's stock
`SpacyRecognizer` (spaCy's generic NER), so anything the generic model
catches that the custom recognizers miss is rendered with **double**
brackets (e.g. `[[PERSON]]`) instead of the normal single-bracket label -
useful for spotting gaps, not meant to be the final redacted output.

```bash
python redact.py --spacy-eval               # both files
python redact.py --text-only --spacy-eval   # text only
```

Output: `data/redacted/encounters.spacy_eval.md`,
`data/redacted/Patient_Intake.spacy_eval.jpg`.

### Clinical-transformer NER evaluation (`.venv-clinical`)

Same idea, but backed by `obi/deid_roberta_i2b2` instead of generic spaCy
NER - meaningfully better precision on clinical text, and it produces
real per-entity confidence scores (spaCy's are a flat constant), so a
score threshold is used to filter out low-confidence noise.

```bash
python redact_clinical_eval.py                        # default threshold (0.6)
python redact_clinical_eval.py --score-threshold 0.8   # experiment with other cutoffs
```

The default comes from `CLINICAL_SCORE_THRESHOLD` in `redact/engine.py`;
pass `--score-threshold` to try other values without editing code.

Output: `data/redacted/encounters.clinical_eval.md`,
`data/redacted/Patient_Intake.clinical_eval.jpg`.

## Design notes

- **Why custom recognizers instead of generic NER**: this data is
  heavily OCR-damaged clinical text (garbled ICD codes, mangled field
  labels, inconsistent punctuation). Generic NER (spaCy or otherwise)
  produces a lot of noise on it (ICD codes tagged as PERSON, procedure
  codes as ORGANIZATION, etc.), so detection here is done with regex
  patterns anchored to known field labels (`DOB:`, `Acc No`, `Home:`,
  `Progress Notes:`, `signed by`, ...) instead.
- **Bracket convention**: single brackets (`[LABEL]`) are the real,
  production redaction. Double brackets (`[[LABEL]]`) only ever appear in
  `--spacy-eval` / `redact_clinical_eval.py` output, marking something an
  NER model caught that the custom recognizers didn't - never mix eval
  output into a document you're actually sharing.
- **Entity-name collisions**: eval-only NER results are namespaced
  internally before anonymizing (`EVAL_ONLY_PREFIX` in
  `redact/engine.py`), so a raw NER label that happens to share a name
  with a production entity type (e.g. both use `"AGE"`) can never
  accidentally borrow the other's anonymizer operator.
