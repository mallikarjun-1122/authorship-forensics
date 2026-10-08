import re


def clean_text(text: str) -> str:
    """Normalize line endings and collapse excessive whitespace while preserving paragraphs."""
    if not text:
        return ""
    # Normalize Windows and Mac carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse horizontal whitespace
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    normalized = "\n".join(lines)
    # Reduce multiple blank lines to double newlines (standard paragraph boundary)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()
