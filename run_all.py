"""
Single entrypoint: reproduce the whole pipeline in one command.

  python run_all.py            # generator check -> tests -> train -> evaluate -> ablation
                               #   -> multiseed -> figures, and write a checkpoint manifest
  python run_all.py --quick    # skip the slow multiseed pass

Each stage is a module already runnable on its own (see README); this just chains them and records
a manifest so the committed numbers are auditable without retraining.
"""

import json
import os
import subprocess
import sys
import time


def sh(mod, *args):
    print(f"\n=== {mod} {' '.join(args)} ===", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", mod, *args])
    print(f"    ({mod} took {time.time()-t0:.0f}s, exit {r.returncode})")
    if r.returncode != 0:
        raise SystemExit(f"{mod} failed")


def write_manifest(ckpt_dir="checkpoints"):
    """Record seed/epochs/best-val/git-hash per checkpoint so numbers are auditable."""
    import torch
    try:
        git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git = "unknown"
    manifest = {"git_hash": git, "checkpoints": {}}
    for fn in sorted(os.listdir(ckpt_dir)):
        if fn.endswith(".pt"):
            blob = torch.load(os.path.join(ckpt_dir, fn), weights_only=False)
            manifest["checkpoints"][fn] = {
                "name": blob.get("name"),
                "best_val": blob.get("best_val"),
                "best_epoch": blob.get("best_epoch"),
                "obs_noise": blob.get("obs_noise", 0.0),
            }
    with open(os.path.join(ckpt_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n=== wrote {ckpt_dir}/manifest.json (git {git}, {len(manifest['checkpoints'])} checkpoints) ===")


def main():
    quick = "--quick" in sys.argv
    sh("lwm.generator")
    sh("lwm.test_invariants")
    sh("lwm.train", "40")
    write_manifest()
    sh("lwm.evaluate")
    sh("lwm.ablation")
    if not quick:
        sh("lwm.multiseed")
    sh("lwm.figures")
    print("\nAll stages complete.")


if __name__ == "__main__":
    main()
