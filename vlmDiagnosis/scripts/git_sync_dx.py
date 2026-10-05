"""
Strict EViCT-Dx GitHub synchronization utility.

Only paths under vlmDiagnosis/ are staged and committed. Frozen EViCT-Core
paths are never staged by this script.

Kaggle usage:
    !python <repo>/vlmDiagnosis/scripts/git_sync_dx.py "message"

GitHub PAT is read from Kaggle Secret: pushEviCT
"""

from pathlib import Path
from kaggle_secrets import UserSecretsClient
import base64
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DX_REL = "vlmDiagnosis"
GITHUB_USERNAME = "itsCodeBakery"
GITHUB_URL = "https://github.com/itsCodeBakery/EviCT.git"
SECRET_NAME = "pushEviCT"
MAX_DIRECT_GIT_FILE_MB = 90


def command(cmd, cwd=ROOT, check=True):
    r = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(cmd)}")
    return r


token = UserSecretsClient().get_secret(SECRET_NAME)
if not token:
    raise RuntimeError(f"Kaggle Secret {SECRET_NAME!r} is unavailable.")

auth = base64.b64encode(f"{GITHUB_USERNAME}:{token}".encode()).decode()
header = f"AUTHORIZATION: basic {auth}"


def git(*args, authenticated=False, check=True):
    cmd = ["git"]
    if authenticated:
        cmd += ["-c", f"http.extraHeader={header}"]
    cmd += list(args)
    return command(cmd, check=check)


if not (ROOT / ".git").exists():
    raise RuntimeError(f"{ROOT} is not a Git checkout.")

# Local identity only.
if not git("config", "--get", "user.name", check=False).stdout.strip():
    git("config", "user.name", GITHUB_USERNAME)
if not git("config", "--get", "user.email", check=False).stdout.strip():
    git("config", "user.email", f"{GITHUB_USERNAME}@users.noreply.github.com")

# Never persist credentials in remote URL.
git("remote", "set-url", "origin", GITHUB_URL)

branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
if branch == "HEAD":
    raise RuntimeError("Detached HEAD: checkout a writable branch before syncing.")

# Stage ONLY the diagnostic extension.
git("add", "-A", "--", DX_REL)

staged = git(
    "diff", "--cached", "--name-only", "--diff-filter=ACMR"
).stdout.splitlines()

bad = [p for p in staged if not (p == DX_REL or p.startswith(DX_REL + "/"))]
if bad:
    raise RuntimeError(
        "Isolation violation: staged path outside vlmDiagnosis/: " + ", ".join(bad)
    )

too_large = []
for rel in staged:
    p = ROOT / rel
    if p.exists() and p.is_file():
        mb = p.stat().st_size / (1024 ** 2)
        if mb > MAX_DIRECT_GIT_FILE_MB:
            too_large.append((rel, mb))
if too_large:
    for rel, mb in too_large:
        print(f"{mb:8.2f} MB  {rel}")
    raise RuntimeError(
        "Refusing ordinary Git commit for >90 MB file. Use a GitHub Release asset."
    )

status = git("diff", "--cached", "--name-only").stdout.strip()
if not status:
    print("✓ vlmDiagnosis already synchronized.")
    sys.exit(0)

message = sys.argv[1] if len(sys.argv) > 1 else "EViCT-Dx Kaggle progress"
git("commit", "-m", message)
print("✓ Created EViCT-Dx-only local commit.")

r = git("push", "origin", branch, authenticated=True, check=False)
if r.returncode != 0:
    combined = (r.stdout + "\n" + r.stderr).lower()
    print(r.stdout[-4000:])
    print(r.stderr[-4000:])
    if (
        "non-fast-forward" in combined
        or "fetch first" in combined
        or "tip of your current branch is behind" in combined
    ):
        print(
            "⚠ Remote branch advanced. Local EViCT-Dx commit retained; push deferred. "
            "Durable run snapshots/checkpoints remain the recovery source."
        )
        sys.exit(0)
    raise RuntimeError("GitHub push failed; local EViCT-Dx commit is retained.")

print("✓ GitHub push successful.")
print(f"✓ Branch: {branch}")
print("✓ Isolation: only vlmDiagnosis/ was staged.")
