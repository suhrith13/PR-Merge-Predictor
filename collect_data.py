"""
PR Merge Predictor — Data Collection Script

Pulls historical, CLOSED pull requests from a list of GitHub repos and
extracts features that would have been available at the moment the PR
was OPENED (not features that only exist after review, e.g. comment
sentiment, number of review rounds, etc). This avoids label leakage.

Label: merged (1) vs closed-without-merge (0)

Usage:
    export GITHUB_TOKEN=ghp_xxxxxxxxxxxx
    python collect_data.py --repos repos.txt --out ../data/prs.csv --max-per-repo 500

repos.txt format (one per line):
    facebook/react
    microsoft/vscode
    ...
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone

import requests

GITHUB_API = "https://api.github.com"


def gh_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def get_with_retry(url: str, headers: dict, params: dict = None, max_retries: int = 5):
    """GET with basic rate-limit-aware retry."""
    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code == 200:
            return resp
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            sleep_for = max(reset - time.time(), 5)
            print(f"  rate limited, sleeping {sleep_for:.0f}s...", file=sys.stderr)
            time.sleep(sleep_for)
            continue
        if resp.status_code == 404:
            return resp
        if resp.status_code == 422:
            print(f"  422 validation error for {url} (this repo/query is likely blocked, skipping): {resp.text[:200]}", file=sys.stderr)
            return resp
        print(f"  unexpected status {resp.status_code} for {url}: {resp.text[:300]}", file=sys.stderr)
        time.sleep(2 ** attempt)
    return resp


def fetch_closed_prs(owner: str, repo: str, headers: dict, max_prs: int):
    """Paginate through closed PRs for a repo, newest first."""
    prs = []
    page = 1
    per_page = 100
    while len(prs) < max_prs:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls"
        params = {
            "state": "closed",
            "sort": "created",
            "direction": "desc",
            "per_page": per_page,
            "page": page,
        }
        resp = get_with_retry(url, headers, params)
        if resp.status_code != 200:
            print(f"  failed to fetch page {page} for {owner}/{repo}: {resp.status_code}", file=sys.stderr)
            break
        batch = resp.json()
        if not batch:
            break
        prs.extend(batch)
        page += 1
        if len(batch) < per_page:
            break
    return prs[:max_prs]


def fetch_contributor_history(owner: str, repo: str, username: str, headers: dict,
                               before_date: str, cache: dict):
    """
    Compute this contributor's PAST merge rate on this repo, using only PRs
    they opened before `before_date`. Cached per (owner, repo, username) to
    avoid redundant calls when a contributor has multiple PRs in the dataset.
    """
    cache_key = (owner, repo, username)
    if cache_key in cache:
        past_prs = cache[cache_key]
    else:
        url = f"{GITHUB_API}/search/issues"
        params = {
            "q": f"repo:{owner}/{repo} is:pr author:{username}",
            "per_page": 100,
        }
        resp = get_with_retry(url, headers, params)
        past_prs = resp.json().get("items", []) if resp.status_code == 200 else []
        cache[cache_key] = past_prs

    prior = [p for p in past_prs if p.get("created_at", "") < before_date and p.get("state") == "closed"]
    if not prior:
        return None  # unknown / first-time contributor signal handled separately

    merged_count = sum(1 for p in prior if p.get("pull_request", {}).get("merged_at"))
    return merged_count / len(prior)


def extract_features(pr: dict, owner: str, repo: str, headers: dict,
                      contributor_cache: dict, repo_baseline: float) -> dict:
    created_at = pr["created_at"]
    closed_at = pr["closed_at"]
    merged_at = pr.get("merged_at")

    created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    closed_dt = datetime.fromisoformat(closed_at.replace("Z", "+00:00")) if closed_at else None

    # Need the PR "files changed" stats — requires a second call to the PR detail endpoint
    pr_number = pr["number"]
    detail_url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}"
    detail_resp = get_with_retry(detail_url, headers)
    detail = detail_resp.json() if detail_resp.status_code == 200 else {}

    additions = detail.get("additions", 0)
    deletions = detail.get("deletions", 0)
    changed_files = detail.get("changed_files", 0)

    author = pr["user"]["login"] if pr.get("user") else "unknown"
    author_association = pr.get("author_association", "NONE")  # OWNER/MEMBER/CONTRIBUTOR/FIRST_TIME_CONTRIBUTOR/etc

    body = pr.get("body") or ""
    title = pr.get("title") or ""

    contributor_merge_rate = fetch_contributor_history(
        owner, repo, author, headers, created_at, contributor_cache
    )
    is_first_time = contributor_merge_rate is None

    has_test_keyword = any(
        kw in (title + " " + body).lower() for kw in ["test", "spec", "coverage"]
    )

    label = 1 if merged_at else 0

    return {
        "owner": owner,
        "repo": repo,
        "pr_number": pr_number,
        "created_at": created_at,
        "additions": additions,
        "deletions": deletions,
        "total_diff": additions + deletions,
        "changed_files": changed_files,
        "title_length": len(title),
        "body_length": len(body),
        "has_test_keyword": int(has_test_keyword),
        "author_association": author_association,
        "is_first_time_contributor": int(is_first_time),
        "contributor_merge_rate": contributor_merge_rate if contributor_merge_rate is not None else repo_baseline,
        "day_of_week": created_dt.weekday(),
        "hour_of_day": created_dt.hour,
        "label_merged": label,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", required=True, help="Path to text file, one owner/repo per line")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--max-per-repo", type=int, default=500)
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: set GITHUB_TOKEN environment variable (a GitHub personal access token).", file=sys.stderr)
        sys.exit(1)
    headers = gh_headers(token)

    with open(args.repos) as f:
        repo_list = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    fieldnames = [
        "owner", "repo", "pr_number", "created_at",
        "additions", "deletions", "total_diff", "changed_files",
        "title_length", "body_length", "has_test_keyword",
        "author_association", "is_first_time_contributor",
        "contributor_merge_rate", "day_of_week", "hour_of_day",
        "label_merged",
    ]

    contributor_cache = {}
    total_rows = 0

    with open(args.out, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for repo_full in repo_list:
            owner, repo = repo_full.split("/")
            print(f"Fetching {repo_full}...")
            prs = fetch_closed_prs(owner, repo, headers, args.max_per_repo)
            if not prs:
                continue

            merged_flags = [1 if p.get("merged_at") else 0 for p in prs]
            repo_baseline = sum(merged_flags) / len(merged_flags) if merged_flags else 0.5

            for i, pr in enumerate(prs):
                try:
                    row = extract_features(pr, owner, repo, headers, contributor_cache, repo_baseline)
                    writer.writerow(row)
                    total_rows += 1
                except Exception as e:
                    print(f"  skipping PR #{pr.get('number')}: {e}", file=sys.stderr)
                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(prs)} processed for {repo_full}")

            print(f"  done: {len(prs)} PRs from {repo_full} (repo baseline merge rate: {repo_baseline:.2f})")

    print(f"\nWrote {total_rows} rows to {args.out}")


if __name__ == "__main__":
    main()
