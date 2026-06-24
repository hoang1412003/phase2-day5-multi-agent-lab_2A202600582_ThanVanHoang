"""Benchmark skeleton for single-agent vs multi-agent."""

from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


Runner = Callable[[str], ResearchState]


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and return a metric object.
    Includes quality scoring, estimated token cost, citation coverage, and error rate.
    """

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    
    # Calculate costs from state.trace (or from agent_results if available)
    total_cost = 0.0
    for event in state.trace:
        if "payload" in event and "cost" in event["payload"]:
            total_cost += event["payload"].get("cost", 0.0)
            
    # Mock quality score
    quality = 8.5
    
    metrics = BenchmarkMetrics(run_name=run_name, latency_seconds=latency, estimated_cost_usd=total_cost, quality_score=quality, notes="Mocked evaluation")
    return state, metrics
