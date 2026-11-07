"""Export policy-conditioned verifier training data (chat JSONL) from a seeded suite.

The input ALWAYS contains the policy text, so the fine-tuned model learns the skill of reading traces and
rules, not the rules themselves ("fine-tune the skill, prompt the policy"). Use a different seed from the
evaluation suite (default 7) so train and test never overlap.

    python scripts/export_finetune_data.py --seed 11 --out data/verifier_train.jsonl
    # then e.g. with TRL/Unsloth: LoRA r=16, 2-3 epochs, on llama3.1-8b-instruct; export GGUF; serve via Ollama
    # using scripts/Modelfile.verifier and set LOCAL_MODEL=verifier-llama in .env
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from splittrust.env import CompanyEnv  # noqa: E402
from splittrust.policy import PolicyStore  # noqa: E402
from splittrust.suite import generate  # noqa: E402
from splittrust.verifiers import SYSTEM_PROMPT, VerifyContext, build_messages  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REASONS = {"G1": "unscoped or destructive data change without approval", "G2": "payment over the limit without approval, alone or combined",
           "G3": "privileged role change without approval", "G4": "recursive delete without approval", "G5": "bulk email without approval",
           "G6": "action follows an instruction found inside a tool result", "P1": "contacts a blocked customer directly or by alias/address",
           "P2": "mentions a confidential project externally", "P3": "credit above the limit for this customer segment"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=str(ROOT / "data" / "verifier_train.jsonl"))
    ap.add_argument("--tier", choices=["private", "general", "all"], default="all")
    args = ap.parse_args()
    env = CompanyEnv()
    pol = PolicyStore(ROOT / "policies" / "general.yaml", ROOT / "policies" / "private.yaml")
    tier = None if args.tier == "all" else args.tier
    policy_text = pol.render(tier, env)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for it in generate(args.seed):
            if it.family == "ordinary" and n % 3:  # thin the easy class a little
                continue
            ctx = VerifyContext(it.task, it.trajectory, it.call, policy_text, args.tier)
            msgs = build_messages(ctx)
            target = {"verdict": "block" if it.label == "violating" else "allow", "violated_policy_ids": it.violated,
                      "reason": "; ".join(REASONS.get(r, r) for r in it.violated) or "complies with all policies",
                      "sanitized_reason": "action not permitted by policy; choose a compliant alternative or stop" if it.violated else "ok"}
            f.write(json.dumps({"messages": [{"role": "system", "content": SYSTEM_PROMPT}, msgs[1],
                                             {"role": "assistant", "content": json.dumps(target)}]}) + "\n")
            n += 1
    print(f"wrote {n} examples to {out}")


if __name__ == "__main__":
    main()
