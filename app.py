#!/usr/bin/env python3
"""
Intelligent Data Analysis Agent — CLI entry point.

Usage:
    python app.py <dataset> [--query "..."] [--model qwen2.5:3b]

Examples:
    python app.py data/sales.csv
    python app.py data/customers.csv --query "What drives churn?"
    python app.py data/sales.csv --model llama3.2:3b --query "Show revenue trends"
"""
import argparse
import sys
import asyncio

from engine.async_executor import AsyncAnalysisExecutor
from utils.llm import check_ollama_health, list_available_models
from utils.logger import get_logger
from config import DEFAULT_MODEL

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Intelligent Data Analysis Agent — local AI-powered EDA"
    )
    parser.add_argument("filepath", help="Dataset path (CSV / Excel / JSON / Parquet)")
    parser.add_argument("--query",  "-q", default="",
                        help="Optional natural-language analysis goal")
    parser.add_argument("--model",  "-m", default=DEFAULT_MODEL,
                        help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--list-models", action="store_true",
                        help="List available Ollama models and exit")
    args = parser.parse_args()

    # Health check
    if not check_ollama_health():
        logger.error("Ollama is not reachable at the configured URL. Start with: ollama serve")
        return 1

    if args.list_models:
        models = list_available_models()
        logger.info("Available models: %s", models)
        for m in models:
            print(f"  {m}")
        return 0

    executor = AsyncAnalysisExecutor(
        filepath=args.filepath,
        user_query=args.query,
        model=args.model,
    )
    result = asyncio.run(executor.run())

    logger.info("=" * 60)
    logger.info("Analysis Complete")
    logger.info("Report : %s", result.get("report"))
    logger.info("Plots  : %d generated", len(result.get("plots", [])))
    failed = [e for e in result.get("execution_log", []) if e["status"] == "failed"]
    if failed:
        logger.warning("Failed steps: %s", [e["step"] for e in failed])
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
