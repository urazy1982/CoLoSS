"""Chat formatting without torch (unit-testable). Re-exported by pilot.common."""
from __future__ import annotations

_SYSTEM_ROLE: dict[str, bool] = {}


def system_role_supported(tok) -> bool:
    """Probe once per tokenizer. Only the template's own 'system role not supported' error
    selects the fallback; any other error (a real bug or a broken template) is raised."""
    key = getattr(tok, "name_or_path", None) or str(id(tok))
    if key not in _SYSTEM_ROLE:
        try:
            tok.apply_chat_template([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
                                    tokenize=False, add_generation_prompt=True)
            _SYSTEM_ROLE[key] = True
        except Exception as e:  # jinja2 TemplateError raised by the template's raise_exception()
            if "system" not in str(e).lower() or type(e).__name__ != "TemplateError":
                raise
            _SYSTEM_ROLE[key] = False
            print(f"[chat] {key}: template rejects the system role ({e}); system text is prepended "
                  "to the user turn for every composition", flush=True)
    return _SYSTEM_ROLE[key]


def chat(tok, prompt: str, system: str | None = None) -> str:
    """Some templates (e.g. Gemma-2) reject a system role. Then the system text is prepended to
    the user turn. The choice is fixed per model, so every composition sees identical prompts."""
    if system:
        if system_role_supported(tok):
            return tok.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        prompt = f"{system}\n\n{prompt}"
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)
