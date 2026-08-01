"""
Safety module — three-layer post-generation verification.

Layer 1: NumericGuard  — deterministic rules (0ms, 0 API)
Layer 2: NLIVerifier   — cross-encoder scoring (<100ms, 0 API)
Layer 3: FactChecker   — LLM deep review (~500ms, fallback only)

See README.md in this directory for architecture details.
"""

from src.safety.numeric_guard import NumericGuard
from src.safety.nli_verifier import NLIVerifier
from src.safety.fact_checker import FactChecker
