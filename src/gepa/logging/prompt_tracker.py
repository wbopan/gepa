# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

try:
    import weave

    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False

if WEAVE_AVAILABLE:

    class ProposedPrompt(weave.Object):
        """A proposed prompt candidate (before valset evaluation)."""

        content: dict[str, str]
        parent_ref: str | None = None
        iteration: int = 0
        minibatch_score_before: float = 0.0
        minibatch_score_after: float = 0.0
        accepted: bool = False

    class AcceptedPrompt(weave.Object):
        """An accepted prompt with valset evaluation results."""

        proposed_ref: str  # ref to the ProposedPrompt
        candidate_idx: int
        valset_score: float

else:

    class ProposedPrompt:  # type: ignore[no-redef]
        """A proposed prompt candidate (before valset evaluation)."""

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class AcceptedPrompt:  # type: ignore[no-redef]
        """An accepted prompt with valset evaluation results."""

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
