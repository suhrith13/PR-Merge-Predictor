# PR Merge Predictor

**Predicts whether a GitHub pull request will get merged — and proves whether it was right.**

🔗 **Live demo:** https://74dca136.pr-merge-predictor.pages.dev
🔗 **Live API:** https://pr-merge-predictor.adavisundar.workers.dev

---

## The problem

Plenty of tools guess whether a pull request looks "good." Almost none of them check their own guesses against reality. This one does.

Every prediction this system makes gets logged. A scheduled job re-checks each one automatically once the pull request actually closes on GitHub, and the live accuracy you see on the site is computed from those real, resolved outcomes — not from an offline test set nobody can verify.

## What it does

Paste any public GitHub pull request URL. The model returns:

- A **probability of merge** (0–100%)
- The **top factors** driving that score, in plain language (e.g. "first-time contributor" decreases the odds, "author's merge history here" increases it)
- If the PR has already closed, it tells you immediately whether the prediction was right

No account, no setup, no waiting — the whole flow takes about a second.

## Why this is different

Most "will this get merged" ideas run into the same wall: there's no reliable ground truth to check predictions against. This project sidesteps that entirely, because **every PR eventually resolves to merged or closed-without-merge — automatically, publicly, and for free, via GitHub's own API.** No crowdsourcing, no guessing, no proxy metrics standing in for the real answer.

That means the accuracy numbers on the live site aren't a claim — they're a running tally, computed from real outcomes, updated every time a scored PR closes.

## Current model performance

Trained on historical PRs across 18 popular open-source repositories (React Native, VS Code, Vue, Django, Flask, Rust, Go, Kubernetes, TensorFlow, PyTorch, and others), evaluated on a held-out test set the model never saw during training:

| Metric | Value | What it means |
|---|---|---|
| AUC | 0.923 | 0.5 = random guessing, 1.0 = perfect — this is a strong, real signal |
| Accuracy | 89.4% | vs. 52.8% for the naive "always guess the majority outcome" baseline |
| Precision | 0.888 | Of PRs predicted to merge, ~89% actually did |
| Recall | 0.916 | Of PRs that actually merged, ~92% were correctly predicted to |
| Brier score | 0.094 | Lower is better-calibrated; indicates the predicted probabilities are meaningfully trustworthy, not just a coin flip dressed up as a percentage |

These are offline evaluation numbers. The site's live `/stats` endpoint shows accuracy computed purely from resolved, real-world outcomes since deployment — check it directly at `https://pr-merge-predictor.adavisundar.workers.dev/stats`.

## How it works

```
Historical PR data (GitHub API)
        │
        ▼
Feature extraction (point-in-time only — no data leakage from
after-the-fact review activity)
        │
        ▼
Logistic regression training + honest evaluation against a
naive baseline
        │
        ▼
Weights exported as plain JSON, embedded directly in a
Cloudflare Worker (no ML runtime needed — a logistic
regression is just a dot product and a sigmoid)
        │
        ▼
Live scoring: paste a PR → Worker fetches it from GitHub →
scores it → logs the prediction to a database
        │
        ▼
Scheduled job re-checks open predictions every 6 hours until
they resolve, recording the real outcome
        │
        ▼
Public accuracy dashboard, computed from real resolved outcomes
```

## Features used

All features are restricted to information that would have been available **at the moment the PR was opened** — nothing derived from later review activity, which would otherwise leak the answer into the input.

- Diff size (lines added/deleted, files changed)
- Title and description length, and whether tests/coverage are mentioned
- Author's association with the repo (owner, member, past contributor, first-time contributor)
- The author's historical merge rate on that specific repo
- Day of week and hour the PR was opened

## Tech stack

- **Model:** logistic regression (scikit-learn), chosen deliberately over a heavier model — it's small enough to embed directly as plain numbers in an edge function, fully interpretable, and fast to iterate on
- **Data collection:** Python + GitHub REST/Search APIs
- **Serving:** Cloudflare Workers (JS, no ML runtime dependency)
- **Verification database:** Cloudflare D1 (SQLite at the edge)
- **Scheduling:** Cloudflare Cron Triggers, re-checking unresolved predictions every 6 hours
- **Frontend:** single-file static HTML/CSS/JS, deployed on Cloudflare Pages

## Project structure

```
pr-merge-predictor/
├── scripts/
│   ├── collect_data.py       # Pulls historical PRs, builds labeled dataset
│   ├── train_model.py        # Trains + evaluates the model, exports weights
│   └── repos.txt             # List of repos used for training data
├── model/
│   └── weights.json          # Exported model weights (feeds directly into the Worker)
├── worker/
│   ├── src/
│   │   ├── index.js          # API routes, D1 logging, scheduled outcome resolution
│   │   └── scoring.js        # Feature extraction + inference (plain JS)
│   ├── schema.sql            # D1 table definition
│   └── wrangler.toml
├── web/
│   └── index.html            # Static frontend
└── data/
    └── prs.csv               # Collected training data (generated locally)
```

## Running it yourself

The steps below are the short version. For a full walkthrough with a verification check at every step, see [SETUP.md](./SETUP.md).

```bash
# 1. Collect training data (requires a GitHub personal access token)
cd scripts
pip install requests pandas scikit-learn
export GITHUB_TOKEN=ghp_yourtoken
python collect_data.py --repos repos.txt --out ../data/prs.csv --max-per-repo 500

# 2. Train the model
python train_model.py --data ../data/prs.csv --out ../model/weights.json

# 3. Set up Cloudflare D1 and deploy the Worker
cd ../worker
npm install
npx wrangler login
npx wrangler d1 create pr-merge-predictor-db   # copy the ID into wrangler.toml
npm run db:init:remote
npx wrangler secret put GITHUB_TOKEN
npm run deploy

# 4. Deploy the frontend (update API_BASE_URL in web/index.html first)
cd ../web
npx wrangler pages deploy .
```

## Honest limitations

- **This is a baseline model, not a finished product.** Logistic regression on a fairly compact feature set gets strong results here, but there's real room to grow — gradient boosting, more features, and per-repo fine-tuning are all natural next steps.
- **Live accuracy will be noisy at low sample sizes.** The site itself flags this — with fewer than 30 resolved predictions, treat the live numbers as provisional.
- **First-time contributors don't have a true "unknown" signal** — the model falls back to a repo-level baseline rather than a distinct category, a modeling simplification documented in the code.
- **A small number of very high-traffic repos block search-API access** (an anti-scraping measure on GitHub's side), so training data slightly under-represents the very largest repos.


