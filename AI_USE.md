# AI use disclosure

## Tools used

- **Claude (Anthropic)**, accessed through claude.ai, model `claude-opus-5-5`.

## My role

- **Problem and design.** I chose the problem and wrote the proposal: thin-file credit scoring with alternative data,
  the nested A/B/C feature-set design, a WoE scorecard as baseline and LightGBM as the primary model.
- **Direction and standards.** I set the requirements the work had to meet: the DSA 8401 brief, my proposal, and the
  structure and simplicity of the course's class-practical notebook. I redirected the work when it drifted from these
  (for example, asking for a simpler notebook, and later restarting it in the class-practical style).
- **Questioning the analysis.** I asked for a confusion matrix, checked that every dataset in the proposal was used,
  questioned how the Home Credit data relates to mobile money, and asked how behaviour could be compared with bureau
  data. These led to the confusion matrix, the use of `bureau_balance.csv`, the Home Credit framing and the
  behaviour-vs-bureau comparison.
- **Responding to review.** After feedback that the conclusion overstated the case for first-time borrowers,
  I qualified it in the notebook.
- **My own changes to the notebook.** I edited the wording and structure, improved the data dictionary (first
  non-missing example per column, sorted by missingness) and narrowed the warnings filter so real problems still show.
- **Running and publishing.** I set up my environment, ran the notebook on the full data on my own computer,
  and published the repository on GitHub with automated tests.
- **Decisions I own:** holding `EXT_SOURCE_*` and `AMT_REQ_CREDIT_BUREAU_*` out of sets A and B; defining thin-file as
  "no record in the bureau table"; excluding `CODE_GENDER` from model inputs while auditing fairness on it; the 5:1
  cost ratio; keeping two anonymised example applicants in the repository.

## What AI did

| Area | Nature of assistance |
|---|---|
| Code | AI generated the code at my request: the pipeline (DuckDB feature SQL, WoE scorecard, LightGBM wrapper, training and evaluation), the FastAPI service, Dockerfile, CI workflow and tests, and the first version of the walkthrough notebook. I reviewed, ran and modified it. |
| Analysis | AI ran the pipeline in its environment and suggested follow-up checks (bureau history vs external scores, retraining without age, the head-to-head test, the first-time-borrower check). |
| Writing | AI drafted the README, the report and the presentation slides from the results; I reviewed and edited them. |
| Explanations | AI explained concepts I asked about (class imbalance, leakage, the confusion matrix, Git and GitHub). |
| Debugging | AI found and fixed two leaks during the build (a blank-field count that revealed bureau status to set A, and a training/serving skew) and a reproducibility bug in the train/test split. |

## How outputs were verified

- Every number in the notebook, report and README comes from executed code, not from AI text. I re-ran the notebook myself.
- Feature SQL is covered by hand-computed unit tests; GitHub Actions runs all 16 tests and builds the Docker image on every push.
- Claims were checked against outputs and corrected where they did not match (for example, qualifying the first-time-borrower conclusion).