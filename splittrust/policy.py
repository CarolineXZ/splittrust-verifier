"""Versioned policy store. Rules are tagged `general` (frontier tier) or `private` (local tier).

The store re-reads its YAML files on every `load()`, so editing a rule takes effect on the next tool call.
Private rules can reference CRM data (`data: blocked_customers`), which is rendered into the rule text
for the local verifier only.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .env import CompanyEnv

SEVERITY_WEIGHT = {"S3": 100.0, "S2": 10.0, "S1": 1.0}


@dataclass(frozen=True)
class Rule:
    id: str
    tier: str  # general | private
    severity: str  # S1 | S2 | S3
    text: str
    check: str | None = None
    data: str | None = None
    actor_hint: str | None = None  # non-sensitive sentence sent to the actor when this rule blocks


class PolicyStore:
    def __init__(self, general_path: str | Path, private_path: str | Path):
        self.general_path = Path(general_path)
        self.private_path = Path(private_path)
        self._rules: list[Rule] = []
        self._version = ""
        self.load()

    # ---- loading
    def load(self) -> "PolicyStore":
        rules: list[Rule] = []
        digest = hashlib.sha256()
        for tier, path in (("general", self.general_path), ("private", self.private_path)):
            raw = path.read_text()
            digest.update(raw.encode())
            doc = yaml.safe_load(raw) or {}
            for r in doc.get("rules", []):
                rules.append(Rule(id=r["id"], tier=tier, severity=r.get("severity", "S2"), text=" ".join(r["text"].split()),
                                  check=r.get("check"), data=r.get("data"),
                                  actor_hint=" ".join(r["actor_hint"].split()) if r.get("actor_hint") else None))
        self._rules = rules
        self._version = digest.hexdigest()[:10]
        return self

    @property
    def version(self) -> str:
        return self._version

    def rules(self, tier: str | None = None) -> list[Rule]:
        return [r for r in self._rules if tier is None or r.tier == tier]

    def rule(self, rule_id: str) -> Rule | None:
        return next((r for r in self._rules if r.id == rule_id), None)

    # ---- rendering
    @staticmethod
    def _render_data(name: str, env: CompanyEnv) -> str:
        if name == "blocked_customers":
            lines = []
            for b in env.blocked_customers():
                lines.append(f"  - {b['name']} (customer {b['customer_id']}); emails: {', '.join(b['emails'])}; "
                             f"aliases: {', '.join(b['aliases'])}; also described as: {b['description']}; reason: {b['reason']}")
            return "Blocked-customer list:\n" + ("\n".join(lines) if lines else "  (none: no customer is currently blocked)")
        if name == "confidential_projects":
            lines = [f"  - codename {p['codename']}; aliases: {', '.join(p['aliases'])}; also described as: {p['description']}"
                     for p in env.confidential_projects()]
            return "Confidential projects:\n" + ("\n".join(lines) if lines else "  (none)")
        return ""

    def render(self, tier: str | None, env: CompanyEnv | None = None) -> str:
        """Policy text for a verifier. Private rules include their private data when `env` is given."""
        out = []
        for r in self.rules(tier):
            out.append(f"[{r.id}] (severity {r.severity}) {r.text}")
            if r.data and env is not None:
                out.append(self._render_data(r.data, env))
        return "\n".join(out)

    def severity_of(self, rule_ids: list[str]) -> str:
        sev = [self.rule(i).severity for i in rule_ids if self.rule(i)]
        return max(sev, key=lambda s: SEVERITY_WEIGHT[s]) if sev else "S1"

    # ---- hot-edit helpers used by the demo and tests
    def raw_private_yaml(self) -> str:
        return self.private_path.read_text()

    def write_private_yaml(self, text: str) -> None:
        yaml.safe_load(text)  # raises on invalid YAML before anything is written
        self.private_path.write_text(text)
        self.load()

    def set_blocked(self, env: CompanyEnv, customer_id: str, blocked: bool) -> None:
        """Edit the CRM's blocked list in place (the private data the local tier reads)."""
        lst = env.state["crm"]["blocked_customers"]
        if not blocked:
            env.state["crm"]["blocked_customers"] = [b for b in lst if b["customer_id"] != customer_id]
        elif not any(b["customer_id"] == customer_id for b in lst):
            c = env.customer_by_id(customer_id)
            lst.append({"customer_id": customer_id, "name": c["name"], "emails": [c["email"]], "aliases": [c["name"].split()[0]],
                        "description": "", "reason": "manual hold"})

    def bump_version(self) -> None:
        """Rewrite private.yaml with version+1 so the policy_version hash changes (e.g. after a data edit)."""
        doc: dict[str, Any] = yaml.safe_load(self.private_path.read_text()) or {}
        doc["version"] = int(doc.get("version", 1)) + 1
        self.private_path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))

    def write_private_rule_text(self, rule_id: str, new_text: str) -> None:
        """Rewrite one private rule's text on disk; the next load() picks it up (demo of hot reload)."""
        doc: dict[str, Any] = yaml.safe_load(self.private_path.read_text()) or {}
        for r in doc.get("rules", []):
            if r["id"] == rule_id:
                r["text"] = new_text
        doc["version"] = int(doc.get("version", 1)) + 1
        self.private_path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
