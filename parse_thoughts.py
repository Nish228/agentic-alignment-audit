"""
parse_thoughts.py — Scan a thoughts.log for incomplete chain-of-thought.

Finds turns where STRATEGY= is empty (meaning the model's reasoning was
truncated before it could output structured fields) and logs them to a CSV.

Usage:
    # Single run — creates or appends to incomplete_cot.csv
    python parse_thoughts.py --log runs/baseline_20260318/thoughts.log \
                             --name "baseline" --tax 40 --pressure yes

    # Another run — appends to the same CSV
    python parse_thoughts.py --log runs/harsh_v2_20260319/thoughts.log \
                             --name "harsh_v2" --tax 20 --pressure no

    # Custom output CSV
    python parse_thoughts.py --log runs/test/thoughts.log \
                             --name "test" --tax 50 --pressure yes \
                             --csv my_results.csv
"""

import os
import re
import csv
import argparse


def parse_thoughts_log(log_path: str) -> dict:
    """
    Parse a thoughts.log file and return per-turn data.
    Returns dict with:
        total_turns, incomplete_turns, incomplete_timesteps, completion_rate
    """
    if not os.path.exists(log_path):
        print(f"  ERROR: File not found: {log_path}")
        return None

    text = open(log_path, "r", encoding="utf-8").read()

    # Each turn block looks like:
    # ======...
    # TURN N  |  ...
    # ======...
    # <raw response>
    # ──────...
    # PARSED → action=...  score=...  reasoning=...
    #          strategy=...
    #
    # Split on the TURN header pattern to get one block per turn
    turn_pattern = r"(={50,}\nTURN\s+\d+.*?\n={50,})"
    parts = re.split(turn_pattern, text)

    # Reassemble: each turn = header + body that follows
    turns = []
    for i, part in enumerate(parts):
        if re.match(r"={50,}\nTURN", part):
            body = parts[i + 1] if i + 1 < len(parts) else ""
            turns.append(part + body)

    total_turns = 0
    incomplete_turns = 0
    incomplete_timesteps = []

    for block in turns:
        turn_match = re.search(r"TURN\s+(\d+)", block)
        if not turn_match:
            continue

        timestep = int(turn_match.group(1))
        total_turns += 1

        # Find the PARSED strategy line
        parsed_match = re.search(r"strategy=(.*?)$", block, re.MULTILINE)
        if parsed_match:
            strategy_val = parsed_match.group(1).strip()
            if strategy_val == "" or strategy_val == "None":
                incomplete_turns += 1
                incomplete_timesteps.append(timestep)
        else:
            # No PARSED line at all — definitely incomplete
            incomplete_turns += 1
            incomplete_timesteps.append(timestep)

    completion_rate = ((total_turns - incomplete_turns) / max(total_turns, 1)) * 100

    return {
        "total_turns": total_turns,
        "incomplete_turns": incomplete_turns,
        "incomplete_timesteps": incomplete_timesteps,
        "completion_rate": completion_rate,
    }


def write_csv(csv_path: str, run_name: str, tax: int, has_pressure: bool,
              result: dict):
    """Append a row to the CSV. Creates the file + header if it doesn't exist."""
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "Run Name",
                "Tax %",
                "Pressure",
                "Total Turns",
                "Incomplete Turns",
                "Completion Rate %",
                "Incomplete Timesteps",
            ])

        timesteps_str = ",".join(str(t) for t in result["incomplete_timesteps"])

        writer.writerow([
            run_name,
            tax,
            "Yes" if has_pressure else "No",
            result["total_turns"],
            result["incomplete_turns"],
            f"{result['completion_rate']:.1f}",
            timesteps_str if timesteps_str else "(none)",
        ])


def main():
    parser = argparse.ArgumentParser(
        description="Parse thoughts.log for incomplete chain-of-thought"
    )
    parser.add_argument("--log", type=str, required=True,
                        help="Path to thoughts.log file")
    parser.add_argument("--name", type=str, required=True,
                        help="Run name for the CSV row")
    parser.add_argument("--tax", type=int, required=True,
                        help="Tax percentage for this run")
    parser.add_argument("--pressure", type=str, required=True,
                        choices=["yes", "no"],
                        help="Whether pressure was active")
    parser.add_argument("--csv", type=str, default="incomplete_cot.csv",
                        help="Output CSV path (default: incomplete_cot.csv)")
    args = parser.parse_args()

    print(f"  Parsing: {args.log}")
    result = parse_thoughts_log(args.log)

    if result is None:
        return

    print(f"  Total turns:      {result['total_turns']}")
    print(f"  Incomplete turns: {result['incomplete_turns']}")
    print(f"  Completion rate:  {result['completion_rate']:.1f}%")

    if result["incomplete_timesteps"]:
        preview = result["incomplete_timesteps"][:10]
        suffix = f"... (+{len(result['incomplete_timesteps'])-10} more)" if len(result["incomplete_timesteps"]) > 10 else ""
        print(f"  Incomplete at:    {preview}{suffix}")

    has_pressure = args.pressure.lower() == "yes"
    write_csv(args.csv, args.name, args.tax, has_pressure, result)

    action = "Appended to" if os.path.exists(args.csv) else "Created"
    print(f"  {action}: {args.csv}")


if __name__ == "__main__":
    main()
