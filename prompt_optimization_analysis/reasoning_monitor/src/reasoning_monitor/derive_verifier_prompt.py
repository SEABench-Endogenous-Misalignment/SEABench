"""Generate or revise the semantic-verifier prompt from a fixed specification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SOURCE = Path(__file__).resolve().parent


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prompt", type=Path, required=True)
    parser.add_argument(
        "--model",
        default=os.environ.get("PROMPT_DERIVATION_MODEL", ""),
        help="API model used to write the verifier prompt",
    )
    parser.add_argument(
        "--seed-prompt", type=Path,
        help="Existing prompt to revise; omit to derive a prompt from scratch",
    )
    parser.add_argument(
        "--validation-feedback", type=Path,
        help="Optional validation-only error summary for a revision",
    )
    args = parser.parse_args()
    if not args.model:
        raise RuntimeError("Provide --model or set PROMPT_DERIVATION_MODEL")

    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Install the package with its API dependencies") from error

    specification = (SOURCE / "verifier_prompt_derivation.md").read_text(encoding="utf-8")
    output_schema = json.loads(
        (SOURCE / "verifier_schema.json").read_text(encoding="utf-8")
    )
    parts = [specification, "\n# Required verifier output schema\n", json.dumps(output_schema, indent=2)]
    if args.seed_prompt:
        parts.extend([
            "\n# Existing verifier prompt to revise\n",
            args.seed_prompt.read_text(encoding="utf-8"),
        ])
    if args.validation_feedback:
        parts.extend([
            "\n# Validation-only feedback\n",
            args.validation_feedback.read_text(encoding="utf-8"),
            "\nRevise only general decision criteria supported by this feedback. "
            "Do not copy task-specific wording or identifiers.\n",
        ])
    request = "\n".join(parts)

    result_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"prompt_markdown": {"type": "string", "minLength": 200}},
        "required": ["prompt_markdown"],
    }
    response = OpenAI().responses.create(
        model=args.model,
        input=request,
        text={
            "format": {
                "type": "json_schema",
                "name": "derived_verifier_prompt",
                "strict": True,
                "schema": result_schema,
            }
        },
    )
    if not response.output_text:
        raise RuntimeError("prompt-derivation model returned no output")
    prompt = json.loads(response.output_text)["prompt_markdown"].strip() + "\n"
    args.output_prompt.parent.mkdir(parents=True, exist_ok=True)
    args.output_prompt.write_text(prompt, encoding="utf-8")
    manifest = {
        "derivation_model": args.model,
        "specification_sha256": digest(specification),
        "request_sha256": digest(request),
        "output_prompt_sha256": digest(prompt),
        "seed_prompt_sha256": (
            digest(args.seed_prompt.read_text(encoding="utf-8")) if args.seed_prompt else None
        ),
        "validation_feedback_sha256": (
            digest(args.validation_feedback.read_text(encoding="utf-8"))
            if args.validation_feedback else None
        ),
        "test_data_used": False,
    }
    manifest_path = args.output_prompt.with_suffix(args.output_prompt.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.output_prompt)


if __name__ == "__main__":
    main()
