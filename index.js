import { parsePrUrl, extractFeatures, scoreFeatures } from "./scoring.js";
import weights from "../../model/weights.json";

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...CORS_HEADERS },
  });
}

async function handleScore(request, env) {
  const body = await request.json().catch(() => null);
  if (!body || !body.url) {
    return json({ error: "Send { url: 'https://github.com/{owner}/{repo}/pull/{number}' }" }, 400);
  }

  const parsed = parsePrUrl(body.url);
  if (!parsed) {
    return json({ error: "Could not parse a GitHub PR URL from that input." }, 400);
  }
  const { owner, repo, number } = parsed;

  let features, raw;
  try {
    ({ features, raw } = await extractFeatures(owner, repo, number, env.GITHUB_TOKEN));
  } catch (e) {
    return json({ error: `Failed to fetch PR data: ${e.message}` }, 502);
  }

  const { probability, topFactors } = scoreFeatures(features, weights);

  // Record the prediction for later verification -- this is the entire
  // point of the project. If the PR is already closed, we can resolve it
  // immediately instead of waiting for the cron job.
  const nowIso = new Date().toISOString();
  const alreadyResolved = raw.state === "closed";

  try {
    if (alreadyResolved) {
      await env.DB.prepare(
        `INSERT INTO predictions (owner, repo, pr_number, pr_url, predicted_score, features_json, created_at, resolved, actual_merged, resolved_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
         ON CONFLICT(owner, repo, pr_number) DO NOTHING`
      ).bind(owner, repo, number, body.url, probability, JSON.stringify(features), nowIso, raw.merged ? 1 : 0, nowIso).run();
    } else {
      await env.DB.prepare(
        `INSERT INTO predictions (owner, repo, pr_number, pr_url, predicted_score, features_json, created_at, resolved)
         VALUES (?, ?, ?, ?, ?, ?, ?, 0)
         ON CONFLICT(owner, repo, pr_number) DO NOTHING`
      ).bind(owner, repo, number, body.url, probability, JSON.stringify(features), nowIso).run();
    }
  } catch (e) {
    // Don't fail the whole request just because logging failed
    console.error("Failed to record prediction:", e.message);
  }

  return json({
    pr: body.url,
    author: raw.authorLogin,
    title: raw.title,
    already_resolved: alreadyResolved,
    actual_outcome: alreadyResolved ? (raw.merged ? "merged" : "closed_without_merge") : null,
    predicted_merge_probability: Math.round(probability * 1000) / 1000,
    top_factors: topFactors,
    model_metrics: weights.eval_metrics,
  });
}

async function handleStats(env) {
  const totalsRow = await env.DB.prepare(
    `SELECT COUNT(*) as total, SUM(resolved) as resolved_count FROM predictions`
  ).first();

  const resolvedRows = await env.DB.prepare(
    `SELECT predicted_score, actual_merged FROM predictions WHERE resolved = 1`
  ).all();

  const rows = resolvedRows.results || [];
  let correct = 0;
  let brierSum = 0;

  for (const r of rows) {
    const predictedLabel = r.predicted_score >= 0.5 ? 1 : 0;
    if (predictedLabel === r.actual_merged) correct += 1;
    const err = r.predicted_score - r.actual_merged;
    brierSum += err * err;
  }

  const n = rows.length;

  return json({
    total_predictions: totalsRow ? totalsRow.total : 0,
    resolved_predictions: n,
    pending_predictions: (totalsRow ? totalsRow.total : 0) - n,
    live_accuracy: n > 0 ? Math.round((correct / n) * 1000) / 1000 : null,
    live_brier_score: n > 0 ? Math.round((brierSum / n) * 1000) / 1000 : null,
    note: n < 30
      ? "Sample size is still small -- treat these live numbers as provisional until more predictions resolve."
      : null,
    offline_eval_metrics: weights.eval_metrics,
  });
}

// Called on a schedule (see wrangler.toml [triggers]) to check whether
// previously-scored, still-open PRs have closed yet. This is the
// self-verifying feedback loop: no crowdsourcing, no guessing -- GitHub
// tells us the real answer once the PR resolves.
async function resolveOutcomes(env) {
  const unresolved = await env.DB.prepare(
    `SELECT owner, repo, pr_number FROM predictions WHERE resolved = 0 LIMIT 100`
  ).all();

  const rows = unresolved.results || [];
  let resolvedCount = 0;

  for (const row of rows) {
    try {
      const resp = await fetch(
        `https://api.github.com/repos/${row.owner}/${row.repo}/pulls/${row.pr_number}`,
        {
          headers: {
            "Authorization": `Bearer ${env.GITHUB_TOKEN}`,
            "Accept": "application/vnd.github+json",
            "User-Agent": "pr-merge-predictor",
          },
        }
      );
      if (!resp.ok) continue;
      const pr = await resp.json();

      if (pr.state === "closed") {
        await env.DB.prepare(
          `UPDATE predictions SET resolved = 1, actual_merged = ?, resolved_at = ? WHERE owner = ? AND repo = ? AND pr_number = ?`
        ).bind(pr.merged ? 1 : 0, new Date().toISOString(), row.owner, row.repo, row.pr_number).run();
        resolvedCount += 1;
      }
    } catch (e) {
      console.error(`Failed to check ${row.owner}/${row.repo}#${row.pr_number}:`, e.message);
    }
  }

  console.log(`resolveOutcomes: checked ${rows.length}, resolved ${resolvedCount}`);
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: CORS_HEADERS });
    }

    if (url.pathname === "/score" && request.method === "POST") {
      return handleScore(request, env);
    }
    if (url.pathname === "/stats" && request.method === "GET") {
      return handleStats(env);
    }
    if (url.pathname === "/" ) {
      return json({
        service: "pr-merge-predictor",
        endpoints: {
          "POST /score": "{ url: 'https://github.com/{owner}/{repo}/pull/{number}' }",
          "GET /stats": "live, self-verified accuracy metrics",
        },
      });
    }
    return json({ error: "Not found" }, 404);
  },

  async scheduled(event, env, ctx) {
    ctx.waitUntil(resolveOutcomes(env));
  },
};
