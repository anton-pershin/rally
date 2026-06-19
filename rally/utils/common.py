from pathlib import Path


def get_project_path() -> Path:
    return Path(__file__).parent.parent.parent


def get_config_path() -> Path:
    return get_project_path() / "config"


def to_boolean(llm_output: str) -> bool:
    s = llm_output.lower()
    if s.startswith("yes") or s.startswith("true"):
        return True
    elif s.startswith("no") or s.startswith("false"):
        return False

    raise ValueError(f"LLM output is inconsistent with the boolean type: '{s}'")
