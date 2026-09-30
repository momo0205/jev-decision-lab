# Laya Run Provenance and Calibration Audit

**Date:** 2026-09-29
**Status:** Approved for implementation
**Scope:** `03-research/jev-decision-lab`

## Goal

Make the existing Laya calibration result easier to reproduce and interpret without touching the held-out samples or treating a 20-item synthetic calibration split as a general accuracy estimate.

## Evidence found

The saved run `20260929T082350Z-laya-calibration` is explicitly on `calibration`, with dataset SHA-256 `0d93dd32608ea1a3c8b98453e28416d0bb188c6acd5fd7211628132b3cee7042` and model revision `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`. Its 3 high-risk false approvals all belong to the same `API credential rotation` sample family; the other high-risk family in this split had 0/4 false approvals. This is a clustered calibration-set diagnostic, not evidence of generalization or a causal explanation of Laya internals.

During review, an overly broad search accidentally surfaced some current `test` records' input text and expected-label context. No provider was run on that split and no test predictions or metrics were examined, but it can no longer be described as an untouched blind holdout for this analysis. Do not run, tune against, or publish final evaluation claims from that split. Preserve the dataset and its hash as historical provenance; design a fresh holdout only after owner review.

## Design

Add backward-compatible, optional runtime provenance to run manifests and generated reports. Record only Python version, operating-system family, machine architecture, and provider-relevant package versions. For the Laya provider, the package allowlist is `laya`, `torch`, and `transformers`; package names whose metadata is absent are omitted. Record the actual inference device only when the loaded Laya agent exposes it. If a device is unavailable, keep it null rather than infer it from hardware capability.

Do not record hostnames, user names, absolute paths, environment variables, credentials, prompts, or sample text. Old manifests without runtime data must remain readable. The public aggregate JSON may include this bounded provenance because it contains no input or per-sample output.

Add a private research note summarizing the calibration-only error cluster and the holdout exposure. Do not change the dataset, inspect more `test` records, run the `test` split, or publish any test result in this task.

## Alternatives considered

1. **Record only the host's available accelerators.** Rejected: availability is not proof of which device actually executed the model.
2. **Record the provider's selected device plus minimal platform/package provenance.** Recommended: best available evidence while remaining backward-compatible; unknown stays explicit.
3. **Capture full environment output (`pip freeze`, full platform string, and system identifiers).** Rejected: excess detail increases privacy and noise without improving this experiment enough.

## Acceptance criteria

- Existing manifests still validate when `runtime` is absent.
- A new run records privacy-bounded runtime provenance; Laya's actual device is recorded only when exposed by its loaded agent.
- Markdown and public aggregate reports show available runtime provenance without adding sample inputs or raw outputs.
- Tests cover missing optional package metadata, absent device, and report compatibility.
- The calibration audit note states the family-level pattern and the test-split incident without exposing any test sample content.
- No `test` split is run or examined further.
