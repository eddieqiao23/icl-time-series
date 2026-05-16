"""Extract a clean loss-vs-step curve from a slurm log."""
import argparse
import re
import sys


def extract(path):
    """Yield (step, loss) pairs in order seen."""
    pat = re.compile(r"loss\s+([0-9.]+):\s*([0-9]+)%\|.*?\|\s+([0-9]+)/[0-9]+")
    with open(path, errors="replace") as f:
        text = f.read()
    text = text.replace("\r", "\n")
    last_step = -1
    for line in text.split("\n"):
        m = pat.search(line)
        if not m:
            continue
        loss = float(m.group(1))
        step = int(m.group(3))
        if step != last_step:
            yield step, loss
            last_step = step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--every", type=int, default=500)
    args = ap.parse_args()
    rows = list(extract(args.path))
    if not rows:
        print(f"(no loss lines found in {args.path})")
        sys.exit(1)
    print(f"{'step':>6s}  loss")
    # Bucket by step / args.every and take the median loss in each bucket (smooth).
    import statistics
    buckets = {}
    for step, loss in rows:
        b = step // args.every
        buckets.setdefault(b, []).append(loss)
    for b in sorted(buckets):
        center = b * args.every
        med = statistics.median(buckets[b])
        print(f"{center:>6d}  {med:.4f}")


if __name__ == "__main__":
    main()
