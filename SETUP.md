# PR Merge Predictor

Predicts whether a GitHub pull request will get merged, and — the part that
matters — checks itself. Every prediction is logged, and a scheduled job
re-checks it once the PR actually closes. The accuracy numbers on the site
are computed from real, resolved outcomes, not offline test data.

## How it fits together

```
scripts/collect_data.py   -> pulls historical PRs, builds a labeled dataset
scripts/train_model.py    -> trains logistic regression, reports real AUC/precision/recall
model/weights.json        -> exported model (plain numbers, no ML runtime needed)
worker/                   -> Cloudflare Worker: scores live PRs, logs predictions to D1,
                              re-checks outcomes on a cron schedule, serves /stats
web/index.html            -> static frontend (Cloudflare Pages), paste a PR URL, get a score
```

## 1. Get a GitHub personal access token

Go to https://github.com/settings/tokens -> Generate new token (classic) ->
no special scopes needed for public repos, just leave everything unchecked
for read-only public access (or check `public_repo` to be safe). Copy the
token, you'll need it twice: once locally for data collection, once as a
Worker secret for live scoring.

## 2. Collect training data

```bash
cd scripts
pip install requests pandas scikit-learn --break-system-packages
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
python collect_data.py --repos repos.txt --out ../data/prs.csv --max-per-repo 500
```

This will take a while — it's making several API calls per PR (detail +
contributor history) across ~20 repos. Expect it to run for 20-60 minutes
depending on your rate limit. It prints progress per repo as it goes.

Feel free to trim `repos.txt` down to fewer repos for a faster first pass —
even 5-10 repos with a few hundred PRs each is enough to get a real,
evaluable baseline model.

## 3. Train the model

```bash
python train_model.py --data ../data/prs.csv --out ../model/weights.json
```

This prints an honest evaluation against a held-out test set:

```
Naive baseline acc (always predict 'X')  : 0.XXX
Model accuracy                            : 0.XXX
Model AUC                                 : 0.XXX
Model precision                           : 0.XXX
Model recall                              : 0.XXX
Brier score                               : 0.XXX
```

**Read this before you ship it.** If AUC is close to 0.5, the model isn't
finding real signal yet — usually means you need more data, more repos,
or better features (this is normal on a first pass, not a failure).
If model accuracy is barely above the naive baseline accuracy, the model
isn't adding much over just guessing the majority class — don't present
it as more accurate than it is.

This overwrites the placeholder `model/weights.json` with your real,
evaluated model.

## 4. Set up Cloudflare D1 (the verification database)

```bash
cd ../worker
npm install
npx wrangler login
npx wrangler d1 create pr-merge-predictor-db
```

Copy the `database_id` it prints out into `wrangler.toml` (replace
`REPLACE_WITH_YOUR_D1_DATABASE_ID`). Then create the table:

```bash
npm run db:init:remote
```

## 5. Set the GitHub token as a Worker secret

```bash
npx wrangler secret put GITHUB_TOKEN
# paste your token when prompted
```

Do **not** put the token directly in `wrangler.toml` — secrets keep it out
of your repo.

## 6. Deploy the Worker

```bash
npm run deploy
```

Wrangler will print your live URL, something like:
`https://pr-merge-predictor.YOUR_SUBDOMAIN.workers.dev`

## 7. Point the web app at your Worker

Open `web/index.html` and update this line near the top of the `<script>`
block:

```js
const API_BASE_URL = "https://pr-merge-predictor.YOUR_SUBDOMAIN.workers.dev";
```

## 8. Deploy the web app to Cloudflare Pages

Easiest path: push the `web/` folder to a GitHub repo, then in the
Cloudflare dashboard: Pages -> Create a project -> Connect to Git -> pick
the repo -> set build output directory to `web` (or `/` if `web/` is the
repo root) -> deploy. No build step needed, it's a single static file.

## 9. Verify it's actually working end to end

1. Visit your deployed Pages URL
2. Paste a real open PR, e.g. `https://github.com/facebook/react/pull/<some open PR number>`
3. Confirm you get a probability and top factors back
4. Paste an *already closed* PR — it should immediately show "already resolved" with the real outcome, since no waiting is needed
5. Check `GET /stats` on your Worker URL directly in the browser — it should return valid JSON even with zero resolved predictions
6. Wait for the cron to run (every 6 hours) or trigger it manually with `npx wrangler dev --test-scheduled` locally, then check `/stats` again — `resolved_predictions` should start increasing as real outcomes come in

## Local testing before deploying

```bash
cd worker
npx wrangler dev
```

This runs the Worker locally with a local D1 instance (`npm run db:init`
first, without `:remote`, to set up the local DB). Update `API_BASE_URL`
in `web/index.html` to `http://localhost:8787` temporarily and open
`web/index.html` directly in a browser to test the full flow before
deploying anything.

## Honest limitations to keep in mind

- The current model is logistic regression on a fairly small feature set —
  it's a real, evaluable baseline, not a state-of-the-art classifier. Good
  starting point, room to grow (gradient boosting, more features, per-repo
  fine-tuning).
- `contributor_merge_rate` for first-time contributors falls back to a
  repo-level baseline rather than a true unknown — this is a modeling
  simplification, documented in the code.
- Live accuracy numbers will be noisy with a small sample size early on.
  The frontend already shows a caveat when `resolved_predictions < 30`.
