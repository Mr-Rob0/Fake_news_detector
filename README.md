# AI News Verification System

An evidence-based AI verification system that analyzes news stories, extracts testable factual claims, retrieves external web evidence, objectively evaluates sources, and produces explainable claim-level verdicts and overall article assessments.

---

## Complete Verification Pipeline (Phases 1–9)

```text
User Input (Direct News Text OR Article URL)
       ↓
Input Validation & URL Normalization (Phase 2)
       ↓
Article Extraction & Content Cleanup (Phase 3)
       ↓
Factual Claim Extraction & Filtering (Phase 4)
       ↓
Focused Query Generation & Evidence Retrieval (Phase 5 / 5.1)
       ↓
Source Evaluation, Provenance & Independence Control (Phase 6 / 8)
       ↓
Domain-Agnostic Generic Claim Representation & Stance Analysis (Phase 9)
       ↓
Explainable Claim Verdict & Verification Summary (Phase 9)
       ↓
Overall Article Assessment (Likely Supported / Likely Contradicted / Mixed / Uncertain)
```

---

## Phase 9: Generic Claim–Evidence Stance Analysis & Explainable Verdict Engine

Phase 9 implements a strictly **domain-agnostic** factual comparison engine in [`src/claim_analyzer.py`](src/claim_analyzer.py) that works seamlessly across arbitrary news domains (politics, sports, science, technology, economy, crime, international affairs, disasters, health, and government announcements):

1. **Generic Claim Representation:**
   - Decomposes assertions into entities, directional action verbs, numbers/percentages, calendar dates/years, negation polarity, and attribution markers.
   - Zero hardcoding of politicians, organizations, protests, or news topics.

2. **Topic Overlap vs Predicate Matching:**
   - Explicitly rejects naive keyword overlap: a source merely mentioning an entity (e.g. quarterly financial results for a company) does not support an uncorroborated event (e.g. quantum chip launch).

3. **Negation & Polarity Handling:**
   - Correctly distinguishes affirmative events from denials, refutations, and grammatical negations.

4. **Attribution & Temporal Reasoning:**
   - Explicitly differentiates unverified allegations or statements ("person X alleged") from established facts.
   - Detects historical mismatches (e.g. 1969 Apollo mission vs 2024 lunar mission) as context rather than current verification.

5. **Quantitative Discrepancy Detection:**
   - Compares exact percentages and metric values against reported evidence.

6. **Explainable Claim Cards & Verification Summary UI:**
   - Streamlit interface presents supporting, contradicting, and neutral evidence blocks with domain provenance badges, URLs, and concise rationale.

---

## Setup & Testing Instructions

### 1. Activate Virtual Environment
```powershell
.venv\Scripts\Activate.ps1
```

### 2. Run Automated Offline Test Suite
All 141 automated unit tests execute completely offline without consuming external API credits or requiring network access:

```powershell
python -m unittest discover tests
```

### 3. Run the Application
```powershell
streamlit run app.py
```

