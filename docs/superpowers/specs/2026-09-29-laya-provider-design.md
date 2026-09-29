# Optional Laya Provider Design

**Date:** 2026-09-29  
**Status:** Approved for implementation  
**Project:** `03-research/jev-decision-lab`

## Goal

Make the already-reviewed Laya multilingual checkpoint reproducible through Jev Decision Lab's existing provider and `DecisionResult` pipeline, without making Laya, PyTorch, or Transformers default dependencies.

## Scope

- Add an optional `laya` provider that uses `convaiinnovations/laya-multilingual` at revision `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67` and Laya `0.3.21`.
- The provider loads the checkpoint once for a run, submits each `RoutingSample.input` with the fixed four-route choice schema and the Chinese language tag `zh`, and maps Laya's typed choice and option probabilities into the existing result contract.
- The CLI exposes `--provider laya`; selecting it without the optional dependency gives an actionable install message. Ordinary CLI imports and CI must not import Laya or Torch.
- First model use may download checkpoint files from Hugging Face; later runs can use the local cache. No Jev or DeepSeek credential is used.
- Laya output is validated; malformed output is recorded as a failed decision and never falls back to rules or another provider. Probabilities are normalized after validation because the SDK rounds values for display.
- Local dollar cost is unknown and stays `null`; zero vendor API charges are not presented as zero total operating cost. Per-sample inference latency excludes checkpoint download and load time.
- Tests and development runs use only `dev` and `calibration`; the held-out `test` split remains untouched until the research protocol is frozen.

## Non-goals

- No DeepSeek-in-Laya hybrid, no Jev-compatible Laya HTTP server, no training/fine-tuning, and no changes to website content.
- No change to the runner protocol or generic `DecisionResult` schema.
- No use of Laya's `act_probability` as confidence or an abstention threshold.

## Acceptance

- The default installation and offline CI continue to work without Laya/PyTorch/Transformers installed.
- Provider tests cover valid mapping, malformed/missing fields, incomplete labels, probability normalization, and sanitized inference failure.
- CLI tests cover provider selection and a clear error when the optional dependency is unavailable.
- README documents install, first-run weight download/cache, invocation, model/revision provenance, and latency/cost interpretation.
