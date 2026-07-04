// scoring.js -- feature extraction + logistic regression inference in plain JS.
// Deliberately dependency-free: a logistic regression is just a dot product
// and a sigmoid, so there's no need for onnxruntime/tfjs/etc in the Worker.

const DEFAULT_CONTRIBUTOR_MERGE_RATE = 0.75; // used when we can't determine history cheaply

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function parsePrUrl(url) {
  // Accepts: https://github.com/{owner}/{repo}/pull/{number}
  const match = url.match(/github\.com\/([^/]+)\/([^/]+)\/pull\/(\d+)/);
  if (!match) return null;
  const [, owner, repo, number] = match;
  return { owner, repo, number: parseInt(number, 10) };
}

async function ghFetch(path, token) {
  const resp = await fetch(`https://api.github.com${path}`, {
    headers: {
      "Authorization": `Bearer ${token}`,
      "Accept": "application/vnd.github+json",
      "User-Agent": "pr-merge-predictor",
      "X-GitHub-Api-Version": "2022-11-28",
    },
  });
  if (!resp.ok) {
    throw new Error(`GitHub API error ${resp.status}: ${await resp.text()}`);
  }
  return resp.json();
}

async function getContributorMergeRate(owner, repo, username, token) {
  try {
    const data = await ghFetch(
      `/search/issues?q=${encodeURIComponent(`repo:${owner}/${repo} is:pr author:${username} is:closed`)}&per_page=100`,
      token
    );
    const items = data.items || [];
    if (items.length === 0) return { rate: null, isFirstTime: true };

    // We don't have merged_at from the search endpoint directly; approximate
    // using pull_request.merged_at when present in the payload.
    const merged = items.filter((i) => i.pull_request && i.pull_request.merged_at).length;
    return { rate: merged / items.length, isFirstTime: false };
  } catch (e) {
    return { rate: null, isFirstTime: true };
  }
}

async function extractFeatures(owner, repo, number, token) {
  const pr = await ghFetch(`/repos/${owner}/${repo}/pulls/${number}`, token);

  const createdAt = new Date(pr.created_at);
  const title = pr.title || "";
  const body = pr.body || "";
  const additions = pr.additions || 0;
  const deletions = pr.deletions || 0;
  const changedFiles = pr.changed_files || 0;
  const authorAssociation = pr.author_association || "NONE";
  const authorLogin = pr.user ? pr.user.login : "unknown";

  const { rate: contributorRate, isFirstTime } = await getContributorMergeRate(
    owner, repo, authorLogin, token
  );

  const hasTestKeyword =
    /\b(test|spec|coverage)\b/i.test(title) || /\b(test|spec|coverage)\b/i.test(body);

  const features = {
    additions,
    deletions,
    total_diff: additions + deletions,
    changed_files: changedFiles,
    title_length: title.length,
    body_length: body.length,
    has_test_keyword: hasTestKeyword ? 1 : 0,
    is_first_time_contributor: isFirstTime ? 1 : 0,
    contributor_merge_rate: contributorRate !== null ? contributorRate : DEFAULT_CONTRIBUTOR_MERGE_RATE,
    day_of_week: createdAt.getUTCDay(),
    hour_of_day: createdAt.getUTCHours(),
    assoc_CONTRIBUTOR: authorAssociation === "CONTRIBUTOR" ? 1 : 0,
    assoc_FIRST_TIME_CONTRIBUTOR: authorAssociation === "FIRST_TIME_CONTRIBUTOR" ? 1 : 0,
    assoc_MEMBER: authorAssociation === "MEMBER" ? 1 : 0,
    assoc_NONE: authorAssociation === "NONE" ? 1 : 0,
    assoc_OWNER: authorAssociation === "OWNER" ? 1 : 0,
  };

  return { features, raw: { state: pr.state, merged: pr.merged, title, authorLogin } };
}

function scoreFeatures(features, weights) {
  const { feature_order, coefficients, intercept, scaler_mean, scaler_scale } = weights;

  let z = intercept;
  const contributions = [];

  for (let i = 0; i < feature_order.length; i++) {
    const name = feature_order[i];
    const rawValue = features[name] ?? 0;
    const scaled = (rawValue - scaler_mean[i]) / (scaler_scale[i] || 1);
    const contribution = scaled * coefficients[i];
    z += contribution;
    contributions.push({ feature: name, value: rawValue, contribution });
  }

  const probability = sigmoid(z);

  // Sort by absolute contribution to build a human-readable explanation
  const topFactors = contributions
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 4)
    .map((c) => ({
      feature: c.feature,
      value: c.value,
      direction: c.contribution >= 0 ? "increases" : "decreases",
    }));

  return { probability, topFactors };
}

export { parsePrUrl, extractFeatures, scoreFeatures };
