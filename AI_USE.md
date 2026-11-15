# AI use disclosure

> **Anne: please review and edit this file so it accurately reflects how you used AI.**
> It is a starting draft written at the time the code was generated, and the course
> policy requires that it be truthful. Delete this note when done.

## Tools used

- **Claude (Anthropic)**, accessed through claude.ai, model `claude-opus-5-5`.

## What the AI assisted with

| Area | Nature of assistance |
|---|---|
| Problem framing & proposal | The proposal (problem, data source, nested A/B/C design, WoE baseline, LightGBM primary model, serving sketch) was written by me before AI assistance on the build. |
| Repository scaffolding | AI generated the initial project layout, Makefile, Dockerfile, docker-compose and CI workflow. |
| Feature engineering code | AI wrote the DuckDB SQL for the behavioural (3/6/12-month windows) and bureau aggregations, implementing the feature families specified in my proposal. |
| Modelling code | AI wrote the WoE binning / scorecard implementation, the LightGBM wrapper, the CV/refit training loop and the evaluation module. |
| API | AI wrote the FastAPI service and its request/response schemas. |
| Tests | AI wrote the unit and API contract tests. |
| Analysis & report | AI ran the pipeline, proposed the follow-up experiments (splitting set C into bureau history vs. external scores; retraining without age), summarised results and drafted the report text. |
| Walkthrough notebook | AI wrote `notebooks/capstone_notebook.ipynb` (and the script that generates it) following the flow of the course's class-practical notebook, at my request. AI also proposed the behaviour-vs-bureau head-to-head test and the confusion matrix, added to both the notebook and the pipeline. |
| Debugging | AI found and fixed two leaks during the build (a blank-field count that revealed bureau status to set A, and a training/serving skew in the same feature), and a reproducibility bug where parallel aggregation changed row order and therefore the train/test split. All are described in the report or README. |

## Design decisions I reviewed and own

- Holding `EXT_SOURCE_*` and `AMT_REQ_CREDIT_BUREAU_*` out of feature set A (they are bureau-derived).
- Defining thin-file as "no record in the bureau table".
- Excluding `CODE_GENDER` from model inputs while auditing fairness on it.
- The 5:1 cost ratio behind the decision threshold.

## How outputs were verified

- Every pipeline stage was run end-to-end on the full data; metrics come from executed code, not from the AI's text.
- Feature SQL is covered by hand-computed unit tests (e.g. late-payment rate, DPD windows, the leakage guard that drops post-application records).
- The Docker image was built and the API smoke-tested with real applicant requests.
