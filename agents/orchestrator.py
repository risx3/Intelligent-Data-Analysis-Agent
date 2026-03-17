"""
Multi-Agent Orchestrator — coordinates all specialist agents.

Agent roster:
  PlannerAgent  → creates analysis plan
  EDAAgent      → parallel data quality + statistics
  VizAgent      → standard + NL-driven visualisations
  InsightAgent  → business insights generation
  CriticAgent   → reflection loop (score + critique)
  ReportGen     → assembles final Markdown report

Flow (with async parallelism):
  load_data
      └─ create_plan
              ├─ EDAAgent.run()         ┐ parallel
              └─ VizAgent.run()         ┘
                      └─ InsightAgent + CriticAgent (reflection loop)
                                  └─ ReportGenerator
"""
import asyncio
import time
from typing import Any

from tools.data_loader import load_data
from agents.planner import create_plan
from agents.eda_agent import EDAAgent
from agents.viz_agent import VizAgent
from agents.insight_agent import generate_insights, regenerate_with_critique
from agents.critic_agent import review_insights, build_eda_summary
from engine.report_generator import build_report
from config import DEFAULT_MODEL, REFLECTION_ENABLED, REFLECTION_MAX_RETRIES


class MultiAgentOrchestrator:
    def __init__(
        self,
        session_id: str | None = None,
        model: str = DEFAULT_MODEL,
    ):
        self.session_id = session_id
        self.model      = model
        self.context: dict[str, Any] = {}
        self.log: list[dict]         = []

    # ── public ─────────────────────────────────────────────────────────────────

    async def run(self, filepath: str, user_query: str = "") -> dict:
        self._print_header(filepath, user_query)

        # 1. Load data (fast, sync)
        await self._step("load_data",
                         asyncio.to_thread(load_data, filepath))

        # 2. Plan
        meta = self._meta()
        await self._step("plan",
                         asyncio.to_thread(create_plan, meta, user_query, self.model))
        plan = self.context["plan"]
        print(f"  Steps : {plan['steps']}")
        if plan.get("analysis_goals"):
            print(f"  Goals : {plan['analysis_goals']}")

        df = self.context["load_data"]["dataframe"]

        # 3. EDA + Viz in parallel
        print("\n[Parallel] EDA + Visualisations...")
        eda_agent = EDAAgent(model=self.model)
        viz_agent = VizAgent(model=self.model)
        eda_result, viz_result = await asyncio.gather(
            self._timed("eda",  eda_agent.run(df, plan.get("focus_columns", []))),
            self._timed("viz",  viz_agent.run(df, user_query, plan)),
        )
        self.context.update(eda_result)
        self.context["generate_visualizations"] = viz_result

        # 4. Insights with optional reflection loop
        print("\n[InsightAgent] Generating insights...")
        insights = await self._run_reflection_loop(user_query)
        self.context["build_insights"] = insights

        # 5. Report
        await self._step("generate_report",
                         asyncio.to_thread(build_report, self.context, user_query))

        return self._build_result()

    # ── reflection loop ────────────────────────────────────────────────────────

    async def _run_reflection_loop(self, user_query: str) -> str:
        meta = self._meta()
        kwargs = dict(
            dataset_meta    = meta,
            describe_result = self.context.get("describe_data", {}),
            missing_result  = self.context.get("check_missing_values", {}),
            stats_result    = self.context.get("run_basic_stats", {}),
            corr_result     = self.context.get("correlation_matrix", {}),
            user_query      = user_query,
            model           = self.model,
        )

        insights = await asyncio.to_thread(generate_insights, **kwargs)

        if not REFLECTION_ENABLED:
            return insights

        eda_summary = build_eda_summary(
            kwargs["dataset_meta"],  kwargs["describe_result"],
            kwargs["missing_result"], kwargs["stats_result"],
            kwargs["corr_result"],
        )

        for attempt in range(1, REFLECTION_MAX_RETRIES + 1):
            critique = await asyncio.to_thread(
                review_insights, insights, eda_summary, self.model
            )
            score   = critique.get("score", 7)
            verdict = critique.get("verdict", "accept")
            print(f"  [Critic] attempt {attempt} — score {score}/10 → {verdict}")

            if verdict == "accept":
                break

            insights = await asyncio.to_thread(
                regenerate_with_critique,
                insights, critique, **kwargs,
            )

        return insights

    # ── helpers ────────────────────────────────────────────────────────────────

    async def _step(self, name: str, coro):
        start = time.time()
        try:
            result = await coro
            self.context[name] = result
            self._record(name, "success", time.time() - start)
        except Exception as exc:
            self.context[name] = {"error": str(exc)}
            self._record(name, "failed", time.time() - start, str(exc))
            print(f"    [!] {name} failed: {exc}")

    async def _timed(self, label: str, coro):
        start = time.time()
        result = await coro
        elapsed = round(time.time() - start, 2)
        print(f"  [✓] {label} ({elapsed}s)")
        return result

    def _record(self, step: str, status: str, duration: float, error: str = ""):
        entry = {"step": step, "status": status, "duration_sec": round(duration, 2)}
        if error:
            entry["error"] = error
        self.log.append(entry)
        sym = "✓" if status == "success" else "✗"
        print(f"  [{sym}] {step} ({entry['duration_sec']}s)")

    def _meta(self) -> dict:
        return {k: v for k, v in self.context.get("load_data", {}).items()
                if k != "dataframe"}

    def _build_result(self) -> dict:
        return {
            "plan":          self.context.get("plan", {}),
            "execution_log": self.log,
            "report":        self.context.get("generate_report", {}).get("report_path"),
            "plots":         self.context.get("generate_visualizations", {}).get("saved_plots", []),
            "insights":      self.context.get("build_insights", ""),
        }

    def _print_header(self, filepath: str, query: str):
        print(f"\n{'='*60}")
        print("  Multi-Agent Data Analysis Orchestrator")
        print(f"  File  : {filepath}")
        if query:
            print(f"  Query : {query}")
        print(f"{'='*60}\n")
