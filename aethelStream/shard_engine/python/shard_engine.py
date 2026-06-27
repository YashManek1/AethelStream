#!/usr/bin/env python3
"""AethelStream M1 — Shard Engine CLI."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .pipeline.ghost_loader import ghost_load
from .pipeline.writer import write_shards
from .pipeline.verifier import verify_index_integrity

def cmd_shard(args: argparse.Namespace) -> int:
    """Shard command: load model, partition, quantize, and write shards."""
    model_id = args.model
    output_dir = Path(args.output)
    print(f"[M1] Ghost-loading {model_id}...")
    _, config, mem_stats = ghost_load(model_id)
    print(f"[M1] {mem_stats.message}")
    if not mem_stats.passed:
        print("[M1] ABORT: ghost load failed memory check", file=sys.stderr)
        return 1
    print(f"[M1] Sharding {model_id} → {output_dir}")
    shard_index, _ = write_shards(model_id, output_dir, num_proc=args.workers)
    print(f"[M1] Written {len(shard_index)} parameter entries")
    verify_index_integrity(shard_index, output_dir)
    return 0

def cmd_verify(args: argparse.Namespace) -> int:
    """Verify command: check shard integrity."""
    index_path = Path(args.output) / "shard_index.json"
    try:
        with open(index_path) as f:
            shard_index = json.load(f)
    except FileNotFoundError:
        print(f"[M1] ABORT: shard_index.json not found at {index_path}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"[M1] ABORT: shard_index.json is malformed: {exc}", file=sys.stderr)
        return 1
    
    try:
        verify_index_integrity(shard_index, args.output)
    except AssertionError as exc:
        print(f"[M1] ABORT: verification failed: {exc}", file=sys.stderr)
        return 1
    return 0

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AethelStream M1 Shard Engine")
    sub = parser.add_subparsers(dest="command")
    p_shard = sub.add_parser("shard", help="Shard a HuggingFace model")
    p_shard.add_argument("model", help="HuggingFace model ID or local path")
    p_shard.add_argument("output", help="Output directory for shards")
    p_shard.add_argument("--workers", type=int, default=4)
    p_verify = sub.add_parser("verify", help="Verify shard integrity")
    p_verify.add_argument("output", help="Shard directory to verify")
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    dispatch = {"shard": cmd_shard, "verify": cmd_verify}
    sys.exit(dispatch[args.command](args))

if __name__ == "__main__":
    main()
