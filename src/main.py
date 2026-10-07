"""Main entry point for the Nepal RAG Agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.utils import setup_logging, get_settings


def cmd_fetch(args: argparse.Namespace) -> None:
    """Run Phase 1: source acquisition."""
    from src.ingestion.fetch_sources import main as fetch_main
    # Re-run with CLI args
    sys.argv = [sys.argv[0]]
    if args.domain:
        sys.argv.extend(["--domain", args.domain])
    if args.config:
        sys.argv.extend(["--config", str(args.config)])
    if args.output_dir:
        sys.argv.extend(["--output-dir", str(args.output_dir)])
    sys.argv.extend(["--delay", str(args.delay)])
    fetch_main()


def cmd_process(args: argparse.Namespace) -> None:
    """Run Phase 2-3: document processing and chunking."""
    from src.processing.pipeline import process_all
    process_all(domain_id=args.domain, verbose=args.verbose)


def cmd_index(args: argparse.Namespace) -> None:
    """Run Phase 4-5: build retrieval index."""
    from src.retrieval.indexer import build_index
    build_index(domain_id=args.domain, force_rebuild=args.force)


def cmd_classify(args: argparse.Namespace) -> None:
    """Train and evaluate domain classifier."""
    from src.domain_classifier.classifier import train_and_evaluate
    train_and_evaluate()


def cmd_chat(args: argparse.Namespace) -> None:
    """Interactive chat session."""
    from src.generation.chat import ChatSession
    settings = get_settings()
    chat = ChatSession(
        provider=args.provider or settings["generation"]["provider"],
        model=args.model or settings["generation"]["model"],
    )
    print("=" * 60)
    print("Nepal Government Services RAG Agent")
    print("Type 'quit' or 'exit' to end, 'reset' to clear history")
    print("=" * 60)
    while True:
        try:
            query = input("\nप्रश्न / Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in ("quit", "exit", "q"):
            break
        if query.lower() == "reset":
            chat.reset()
            print("History cleared.")
            continue
        if not query:
            continue
        response = chat.ask(query)
        print(f"\nउत्तर / Answer:\n{response.answer}")
        if response.citations:
            print(f"\nस्रोत / Sources:")
            for i, cite in enumerate(response.citations, 1):
                print(f"  [{i}] {cite['name']} — {cite['url']}")
        if response.refused:
            print("(Answer refused — insufficient evidence)")


def cmd_ask(args: argparse.Namespace) -> None:
    """Single question-answer."""
    from src.generation.chat import ChatSession
    settings = get_settings()
    chat = ChatSession(
        provider=args.provider or settings["generation"]["provider"],
        model=args.model or settings["generation"]["model"],
    )
    response = chat.ask(args.question)
    print(response.answer)
    if response.citations:
        print(f"\nSources:")
        for i, cite in enumerate(response.citations, 1):
            print(f"  [{i}] {cite['name']} — {cite['url']}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation."""
    from src.evaluation.runner import run_evaluation
    run_evaluation(
        output_path=args.output,
        domains=args.domains.split(",") if args.domains else None,
    )


def cmd_serve(args: argparse.Namespace) -> None:
    """Launch Gradio web interface."""
    from src.app.gradio_app import launch_app
    launch_app(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nepal-rag",
        description="Domain-Aware RAG Agent for Nepal Government Public Services",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--log-file", default=None, help="Log file path")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # fetch
    p_fetch = sub.add_parser("fetch", help="Fetch government sources (Phase 1)")
    p_fetch.add_argument("--domain", default=None, help="Fetch only this domain")
    p_fetch.add_argument("--config", default=None, help="Path to domains.yaml")
    p_fetch.add_argument("--output-dir", default=None, help="Output directory")
    p_fetch.add_argument("--delay", type=float, default=1.0, help="Delay between requests")

    # process
    p_proc = sub.add_parser("process", help="Process and chunk documents (Phase 2-3)")
    p_proc.add_argument("--domain", default=None, help="Process only this domain")
    p_proc.add_argument("--verbose", action="store_true")

    # index
    p_index = sub.add_parser("index", help="Build retrieval index (Phase 4-5)")
    p_index.add_argument("--domain", default=None, help="Index only this domain")
    p_index.add_argument("--force", action="store_true", help="Force rebuild")

    # classify
    sub.add_parser("classify", help="Train domain classifier")

    # chat
    p_chat = sub.add_parser("chat", help="Interactive chat")
    p_chat.add_argument("--provider", default=None)
    p_chat.add_argument("--model", default=None)

    # ask
    p_ask = sub.add_parser("ask", help="Single question")
    p_ask.add_argument("question", help="The question to ask")
    p_ask.add_argument("--provider", default=None)
    p_ask.add_argument("--model", default=None)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Run evaluation")
    p_eval.add_argument("--output", default=None, help="Output file for results")
    p_eval.add_argument("--domains", default=None, help="Comma-separated domain IDs")

    # serve
    p_serve = sub.add_parser("serve", help="Launch web interface")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=7860)
    p_serve.add_argument("--share", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(level=args.log_level, log_file=args.log_file)

    if not args.command:
        parser.print_help()
        return

    commands = {
        "fetch": cmd_fetch,
        "process": cmd_process,
        "index": cmd_index,
        "classify": cmd_classify,
        "chat": cmd_chat,
        "ask": cmd_ask,
        "evaluate": cmd_evaluate,
        "serve": cmd_serve,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
