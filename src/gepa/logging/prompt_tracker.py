# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import weave


class GEPAPrompt(weave.Prompt):
    """A GEPA prompt candidate with optimization metadata."""

    content: dict[str, str]
    iteration: int = 0
    parent_ref: str | None = None

    # Minibatch evaluation scores
    minibatch_score_before: float = 0.0
    minibatch_score_after: float = 0.0

    # Acceptance status and valset evaluation
    accepted: bool = False
    candidate_idx: int | None = None  # Only set when accepted
    valset_score: float | None = None  # Only set when accepted

    def format(self, **kwargs) -> dict[str, str]:
        """Return the prompt content, optionally with parameter substitution."""
        result = {}
        for key, value in self.content.items():
            result[key] = value.format(**kwargs) if kwargs else value
        return result
