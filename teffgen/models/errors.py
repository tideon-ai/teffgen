"""
tideon.ai model error types.

Defines exceptions raised by model adapters beyond the standard
RuntimeError/ValueError hierarchy.
"""

from __future__ import annotations


class ModelRefusalError(Exception):
    """Raised when a model refuses to answer a structured-output request.

    OpenAI structured outputs may return a ``refusal`` field instead of
    valid JSON content.  The adapter raises this exception so callers can
    distinguish a genuine refusal (policy / safety) from a malformed
    response or a network error.

    Attributes:
        refusal_message: The raw refusal string returned by the model.
        model_name: The model that issued the refusal.
    """

    def __init__(self, refusal_message: str, model_name: str = "") -> None:
        self.refusal_message = refusal_message
        self.model_name = model_name
        suffix = f" (model={model_name!r})" if model_name else ""
        super().__init__(f"Model refused to generate structured output{suffix}: {refusal_message}")
