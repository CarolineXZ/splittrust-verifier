"""The gateway sits inside the perimeter between the actor and the tools.

For every mutating call it: reloads policy (hot reload), re-hydrates placeholders to real values,
runs the deterministic backstop, fans out IN PARALLEL to the local verifier (raw view + private rules)
and the frontier verifier (sanitized view + general rules), fails closed on any error, executes only
on unanimous allow, returns a sanitized reason to the actor on block, and escalates after N retries.
Every decision is logged with the policy version that produced it.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from .backstop import Backstop
from .canary import CanaryRegistry, Pseudonymizer
from .env import MUTATING_TOOLS, CompanyEnv, ToolCall, ToolResult
from .policy import PolicyStore
from .verifiers import Verdict, VerifyContext


def asdict_ctx(ctx: VerifyContext) -> dict[str, Any]:
    return {"task": ctx.task, "trajectory": ctx.trajectory, "call": ctx.call.as_text(), "policy": ctx.policy_text}


@dataclass
class Decision:
    call_sanitized: str
    call_raw: str
    executed: bool
    blocked_by: list[str]
    verdicts: dict[str, dict[str, Any]]
    policy_version: str
    latency_ms: float
    reason_to_actor: str
    escalated: bool = False
    result: str = ""
    reason_canaries: list[str] = field(default_factory=list)
    mutating: bool = True


class Gateway:
    def __init__(self, env: CompanyEnv, policies: PolicyStore, backstop: Backstop | None, local: Any, frontier: Any,
                 canaries: CanaryRegistry, pseudonymizer: Pseudonymizer | None = None, max_retries: int = 2,
                 parallel: bool = True, timeout_s: float = 90.0, verify_reads: bool = False,
                 local_rules: str | None = "private", frontier_rules: str | None = "general", frontier_view: str = "sanitized"):
        """`local_rules` / `frontier_rules`: which policy tier each verifier is given (None = all rules).
        `frontier_view`: "sanitized" (placeholders) or "raw" (the privacy-violating upper bound, B3)."""
        self.env, self.policies, self.backstop = env, policies, backstop
        self.local, self.frontier = local, frontier
        self.canaries = canaries
        self.ps = pseudonymizer or Pseudonymizer(env)
        self.max_retries, self.parallel, self.timeout_s, self.verify_reads = max_retries, parallel, timeout_s, verify_reads
        self.local_rules, self.frontier_rules, self.frontier_view = local_rules, frontier_rules, frontier_view
        self.log: list[Decision] = []
        self.frontier_canary_hits: list[dict[str, Any]] = []  # canaries that reached the frontier-bound context
        self._retries: dict[str, int] = {}
        self._pool = ThreadPoolExecutor(max_workers=4)

    # ---- views
    def _views(self, task: str, trajectory_raw: list[dict[str, str]], call_raw: ToolCall):
        """The local tier sees everything; the frontier tier sees only placeholders (unless configured raw)."""
        local_ctx = VerifyContext(task, trajectory_raw, call_raw, self.policies.render(self.local_rules, self.env), "private")
        if self.frontier_view == "raw":
            front_ctx = VerifyContext(task, trajectory_raw, call_raw, self.policies.render(self.frontier_rules, self.env), "general")
        else:
            traj_san = [{"call": self.ps.to_sanitized(t["call"]), "result": self.ps.to_sanitized(t["result"])} for t in trajectory_raw]
            # Private rules are never rendered with private data for the frontier, even when misconfigured to see them.
            front_ctx = VerifyContext(self.ps.to_sanitized(task), traj_san, self.ps.to_sanitized(call_raw),
                                      self.policies.render(self.frontier_rules), "general")
        if self.frontier is not None:
            found = self.canaries.scan(json.dumps(asdict_ctx(front_ctx), ensure_ascii=False))
            if found:
                self.frontier_canary_hits.append({"call": call_raw.as_text(), "canaries": found})
        return local_ctx, front_ctx

    # ---- main entry
    def handle(self, task_id: str, task: str, trajectory_raw: list[dict[str, str]], call_from_actor: ToolCall) -> Decision:
        t0 = time.perf_counter()
        self.policies.load()  # hot reload: an edited rule applies to this very call
        call_raw = self.ps.to_raw(call_from_actor)
        call_san = self.ps.to_sanitized(call_raw)
        if call_from_actor.name not in MUTATING_TOOLS and not self.verify_reads:
            res = self.env.execute(call_raw)
            d = Decision(call_san.as_text(), call_raw.as_text(), True, [], {}, self.policies.version,
                         (time.perf_counter() - t0) * 1000, "", result=res.as_text(), mutating=False)
            self.log.append(d)
            return d

        verdicts: dict[str, Verdict] = {}
        if self.backstop is not None:
            verdicts["backstop"] = self.backstop.check(call_raw)
        if verdicts.get("backstop", Verdict(True)).allow:
            ctx_local, ctx_frontier = self._views(task, trajectory_raw, call_raw)
            jobs = {}
            if self.local is not None:
                jobs["local"] = (self.local, ctx_local)
            if self.frontier is not None:
                jobs["frontier"] = (self.frontier, ctx_frontier)
            if self.parallel and len(jobs) > 1:
                futs = {k: self._pool.submit(v.judge, c) for k, (v, c) in jobs.items()}
                for k, f in futs.items():
                    verdicts[k] = self._collect(k, lambda f=f: f.result(timeout=self.timeout_s))
            else:
                for k, (v, c) in jobs.items():
                    verdicts[k] = self._collect(k, lambda v=v, c=c: v.judge(c))

        blocked_by = [k for k, v in verdicts.items() if not v.allow]
        executed = not blocked_by
        reason = ""
        reason_canaries: list[str] = []
        result = ""
        escalated = False
        if executed:
            result = self.env.execute(call_raw).as_text()
            self._retries.pop(task_id, None)
        else:
            from .policy import SEVERITY_WEIGHT  # noqa: WPS433
            rules = [r for k in blocked_by for r in (self.policies.rule(i) for i in verdicts[k].violated) if r and r.actor_hint]
            rules.sort(key=lambda r: -SEVERITY_WEIGHT[r.severity])
            parts = [r.actor_hint for r in rules]
            if not parts:
                parts = [verdicts[k].sanitized_reason or "action not permitted by policy" for k in blocked_by]
            reason = " ".join(dict.fromkeys(parts))
            reason_canaries = self.canaries.scan(reason)
            if reason_canaries:  # never let private material ride out on the reason channel
                reason = "action not permitted by policy; try a compliant alternative or stop and report"
            n = self._retries.get(task_id, 0) + 1
            self._retries[task_id] = n
            if n > self.max_retries:
                escalated = True
                reason += " [escalated to a human reviewer; do not retry]"
        d = Decision(call_san.as_text(), call_raw.as_text(), executed, blocked_by, {k: asdict(v) for k, v in verdicts.items()},
                     self.policies.version, (time.perf_counter() - t0) * 1000, reason, escalated=escalated, result=result,
                     reason_canaries=reason_canaries)
        self.log.append(d)
        return d

    @staticmethod
    def _collect(name: str, fn) -> Verdict:
        try:
            return fn()
        except Exception as e:  # noqa: BLE001  fail closed
            return Verdict(False, [], f"{name} verifier failed: {e}", tier=name, sanitized_reason="verification failed; action not executed",
                           error="exception")

    # ---- what the actor receives
    @staticmethod
    def to_tool_result(d: Decision) -> ToolResult:
        if d.executed:
            return ToolResult(True, d.result)
        return ToolResult(False, "", blocked=True, reason=d.reason_to_actor)

    def dump_log(self, path: str) -> None:
        with open(path, "w") as f:
            for d in self.log:
                f.write(json.dumps(asdict(d)) + "\n")