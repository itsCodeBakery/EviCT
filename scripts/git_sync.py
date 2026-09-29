"""
EviCT GitHub synchronization utility.

Usage inside Kaggle:

    !python /kaggle/working/EviCT/scripts/git_sync.py "Your commit message"

The GitHub PAT is read securely from the Kaggle Secret:
    pushEviCT

The PAT is never written into the repository remote.
"""

from pathlib import Path
from kaggle_secrets import UserSecretsClient
import subprocess
import base64
import sys
import os


REPO = Path("/kaggle/working/EviCT")

GITHUB_USERNAME = "itsCodeBakery"
GITHUB_URL = "https://github.com/itsCodeBakery/EviCT.git"
SECRET_NAME = "pushEviCT"

# Conservative protection against accidentally trying to commit
# huge binary artifacts directly through ordinary Git.
MAX_DIRECT_GIT_FILE_MB = 90


def command(cmd, cwd=REPO, check=True):
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if check and result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        raise RuntimeError(
            f"Git command failed: return code {result.returncode}"
        )

    return result


token = UserSecretsClient().get_secret(SECRET_NAME)

if not token:
    raise RuntimeError(
        f"Kaggle Secret '{SECRET_NAME}' is unavailable."
    )

auth = base64.b64encode(
    f"{GITHUB_USERNAME}:{token}".encode()
).decode()

header = f"AUTHORIZATION: basic {auth}"


def git(*args, authenticated=False, check=True):

    cmd = ["git"]

    if authenticated:
        cmd += [
            "-c",
            f"http.extraHeader={header}",
        ]

    cmd += list(args)

    return command(cmd, check=check)


# Ensure repository-local Git identity exists in fresh Kaggle runtimes.
# This is intentionally local to the EviCT checkout; it does not modify the
# notebook user's global Git configuration.
name = git("config", "--get", "user.name", check=False).stdout.strip()
email = git("config", "--get", "user.email", check=False).stdout.strip()

if not name:
    git("config", "user.name", GITHUB_USERNAME)

if not email:
    git("config", "user.email", f"{GITHUB_USERNAME}@users.noreply.github.com")

# Ensure remote contains no secret.
git("remote", "set-url", "origin", GITHUB_URL)

branch = git(
    "rev-parse",
    "--abbrev-ref",
    "HEAD"
).stdout.strip()

# Stage everything allowed by .gitignore.
git("add", "-A")


# ------------------------------------------------------------
# Large-file safety check
# ------------------------------------------------------------

staged = git(
    "diff",
    "--cached",
    "--name-only",
    "--diff-filter=ACMR"
).stdout.splitlines()

too_large = []

for relative_path in staged:

    path = REPO / relative_path

    if path.exists() and path.is_file():

        size_mb = path.stat().st_size / (1024 ** 2)

        if size_mb > MAX_DIRECT_GIT_FILE_MB:
            too_large.append(
                (relative_path, size_mb)
            )

if too_large:

    print("\nLarge files detected:")
    for name, size in too_large:
        print(f"  {size:8.2f} MB  {name}")

    raise RuntimeError(
        "\nRefusing ordinary Git commit for large files.\n"
        "These files require the EviCT large-artifact "
        "backup workflow instead."
    )


# ------------------------------------------------------------
# Commit only if something changed
# ------------------------------------------------------------

status = git(
    "status",
    "--porcelain"
).stdout.strip()

if not status:

    print("✓ Repository already synchronized.")
    sys.exit(0)


message = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "EviCT Kaggle checkpoint"
)

git(
    "commit",
    "-m",
    message,
)

print("✓ Local Git commit created.")


# ------------------------------------------------------------
# Push using temporary authentication header
# ------------------------------------------------------------

result = git(
    "push",
    "origin",
    branch,
    authenticated=True,
    check=False,
)

if result.returncode != 0:

    print(result.stdout[-4000:])
    print(result.stderr[-4000:])

    combined = (result.stdout + "\n" + result.stderr).lower()

    # Notebook 06 may run for hours while the remote main branch advances
    # because execution-safety fixes are committed from another client. In
    # that specific case the scientific checkpoint/release workflow remains
    # authoritative, and blocking training on a metadata-only non-fast-forward
    # push is harmful. Keep the local commit and continue. A later reconciliation
    # can rebase/push the small metadata commits after GPU work is finished.
    if (
        "non-fast-forward" in combined
        or "fetch first" in combined
        or "tip of your current branch is behind" in combined
    ):
        print(
            "⚠ GitHub main advanced; metadata push deferred. "
            "Local commit retained. Training/release durability continues."
        )
        sys.exit(0)

    raise RuntimeError(
        "\nGitHub push failed for a reason other than non-fast-forward.\n"
        "The local commit is still safe in "
        "/kaggle/working/EviCT.\n"
        "Do NOT delete the Kaggle session before resolving it."
    )


print("✓ GitHub push successful.")
print(f"✓ Branch: {branch}")
print(f"✓ Repository: {GITHUB_URL}")
