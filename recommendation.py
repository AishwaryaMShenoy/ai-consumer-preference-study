import pandas as pd
import numpy as np

ATTRIBUTES = ["price", "battery_hours", "anc", "comfort", "sound", "weight", "design"]


def minmax_normalize(series, higher_is_better=True):
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if hi == lo:
        out = pd.Series(np.ones(len(s)), index=s.index)
    else:
        out = (s - lo) / (hi - lo)
    return out if higher_is_better else 1 - out


def normalized_products(products: pd.DataFrame) -> pd.DataFrame:
    p = products.copy()
    p["price_n"] = minmax_normalize(p["price"], higher_is_better=False)
    p["battery_hours_n"] = minmax_normalize(p["battery_hours"])
    for a in ["anc", "comfort", "sound", "weight", "design"]:
        p[f"{a}_n"] = minmax_normalize(p[a])
    p["review_n"] = minmax_normalize(p["review_rating"])
    return p


def rank_products(
    products: pd.DataFrame,
    weights: dict,
    concerns=None,
    budget_max=None,
    budget_strict=False,
    top_n=None,
) -> pd.DataFrame:
    """Deterministic ranking from the participant's current preference profile.

    The LLM does not directly choose a product. It extracts the participant's
    current priorities, concerns, and explicit budget constraint; this function
    applies those inputs consistently to the catalog.
    """
    p = normalized_products(products)
    weights = normalize_weights(weights)
    concerns = concerns or {}

    score = np.zeros(len(p), dtype=float)

    # Preference-based score
    for a, w in weights.items():
        col = f"{a}_n"
        if col in p:
            score += float(w) * p[col].to_numpy()

    # Review quality is a small general signal.
    score += 0.08 * p["review_n"].to_numpy()

    # ---------------------------------------------------------
    # Concern penalties
    # ---------------------------------------------------------
    penalties = np.zeros(len(p), dtype=float)

    concern_map = {
        "battery_durability": {
            "H01": 0.18,
        },
        "frequent_charging": {
            "H06": 0.16,
            "H10": 0.07,
        },
        "noise_isolation": {
            "H01": 0.12,
            "H02": 0.10,
            "H03": 0.07,
        },
        "calls": {
            "H01": 0.10,
            "H03": 0.09,
            "H09": 0.10,
        },
        "weight": {
            "H04": 0.10,
            "H08": 0.07,
            "H10": 0.15,
        },
        "comfort": {
            "H07": 0.09,
        },
        "sound_natural": {
            "H07": 0.12,
        },
        "touch_controls": {
            "H05": 0.08,
        },
        "bass": {
            "H07": 0.07,
        },
        "battery_with_anc": {
            "H08": 0.10,
        },
        "durability": {
            "H01": 0.15,
            "H06": 0.06,
        },
        "price_value": {
            "H04": 0.06,
            "H06": 0.04,
            "H10": 0.07,
        },
    }

    for concern, strength in concerns.items():
        if not strength:
            continue

        for pid, penalty in concern_map.get(concern, {}).items():
            idx = p.index[p["product_id"] == pid]

            if len(idx):
                penalties[p.index.get_loc(idx[0])] += float(strength) * penalty

    # ---------------------------------------------------------
    # Budget constraint
    # ---------------------------------------------------------
    budget_violation = np.zeros(len(p), dtype=int)
    budget_penalty = np.zeros(len(p), dtype=float)

    if budget_max is not None and float(budget_max) > 0:
        budget_max = float(budget_max)

        excess_ratio = np.maximum(
            0,
            (p["price"].astype(float).to_numpy() - budget_max) / budget_max
        )

        budget_violation = (p["price"].astype(float).to_numpy() > budget_max).astype(int)

        if budget_strict:
            # A strict ceiling is treated as a real constraint.
            # Over-budget products remain visible but are ranked below
            # products that satisfy the ceiling.
            budget_penalty = 0.80 + np.minimum(excess_ratio, 1.0) * 0.20

            # Products within budget receive no budget penalty.
            budget_penalty = np.where(
                budget_violation == 1,
                budget_penalty,
                0.0
            )

        else:
            # A flexible budget is a preference rather than a hard constraint.
            # Going slightly over budget is possible, but carries a penalty.
            budget_penalty = np.minimum(excess_ratio * 0.35, 0.35)

    p["recommendation_score"] = (
        score
        - penalties
        - budget_penalty
    )

    p["tradeoff_penalty"] = penalties
    p["budget_penalty"] = budget_penalty
    p["budget_violation"] = budget_violation

    # Strict budgets are true constraints:
    # feasible products are ranked before products above the ceiling.
    if budget_strict and budget_max is not None and float(budget_max) > 0:
        p = p.sort_values(
            ["budget_violation", "recommendation_score", "product_id"],
            ascending=[True, False, True],
        )
    else:
        p = p.sort_values(
            ["recommendation_score", "product_id"],
            ascending=[False, True],
        )

    p = p.reset_index(drop=True)
    p["rank"] = np.arange(1, len(p) + 1)

    return p if top_n is None else p.head(top_n)
    for concern, strength in concerns.items():
        if not strength:
            continue
        for pid, penalty in concern_map.get(concern, {}).items():
            idx = p.index[p["product_id"] == pid]
            if len(idx):
                penalties[p.index.get_loc(idx[0])] += float(strength) * penalty

    p["recommendation_score"] = score - penalties
    p["tradeoff_penalty"] = penalties
    p = p.sort_values(["recommendation_score", "product_id"], ascending=[False, True]).reset_index(drop=True)
    p["rank"] = np.arange(1, len(p) + 1)
    return p if top_n is None else p.head(top_n)


def normalize_weights(weights: dict) -> dict:
    clean = {k: max(0.0, float(v)) for k, v in weights.items() if k in ATTRIBUTES}
    for k in ATTRIBUTES:
        clean.setdefault(k, 0.0)
    total = sum(clean.values())
    if total <= 0:
        return {k: 1 / len(ATTRIBUTES) for k in ATTRIBUTES}
    return {k: v / total for k, v in clean.items()}


def preference_strength(weights: dict) -> float:
    w = np.array(list(normalize_weights(weights).values()), dtype=float)
    return float(np.sum(w ** 2))


def top3_overlap(baseline_top3, post_top3):
    return len(set(baseline_top3) & set(post_top3))
