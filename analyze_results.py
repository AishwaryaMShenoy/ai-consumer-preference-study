from pathlib import Path

import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

RESULTS = Path("data/participants.csv")
OUT = Path("data/analysis_summary.csv")


def main():
    if not RESULTS.exists():
        raise SystemExit("No data found. Run the Streamlit study and collect responses first.")

    df = pd.read_csv(RESULTS)
    print("\nRows:", len(df))
    print("\nGroup counts:")
    print(df["group"].value_counts())

    print("\nChoice change by group:")
    print(df.groupby("group")["choice_change_count"].agg(["count", "mean", "std"]))

    ai = df.loc[df.group == "AI", "choice_change_count"].dropna()
    control = df.loc[df.group == "CONTROL", "choice_change_count"].dropna()
    if len(ai) >= 2 and len(control) >= 2:
        t, p = stats.ttest_ind(ai, control, equal_var=False)
        print(f"\nWelch t-test, AI vs CONTROL on number of changed top-3 choices: t={t:.3f}, p={p:.4f}")

    print("\nTop-1 choice change by group:")
    print(pd.crosstab(df["group"], df["top1_changed"], normalize="index"))

    print("\nMean top-3 overlap by group:")
    print(df.groupby("group")["top3_overlap_count"].agg(["count", "mean", "std"]))

    print("\nBaseline preference strength by group:")
    print(df.groupby("group")["baseline_strength"].agg(["count", "mean", "std"]))

    # Exploratory moderation: does baseline preference strength relate to choice change differently by condition?
    try:
        model = smf.ols(
            "choice_change_count ~ C(group) * baseline_strength + C(ai_use_frequency)",
            data=df,
        ).fit()
        print("\nExploratory moderation/regression:")
        print(model.summary())
    except Exception as e:
        print("\nRegression could not be fitted:", e)

    try:
        print("\nCorrelation: trade-off reconsideration vs choice change")
        print(df[["tradeoff_reconsideration", "choice_change_count"]].corr())
    except Exception:
        pass

    summary = df.groupby("group").agg(
        participants=("participant_id", "count"),
        mean_choice_change=("choice_change_count", "mean"),
        mean_top3_overlap=("top3_overlap_count", "mean"),
        mean_baseline_strength=("baseline_strength", "mean"),
        mean_tradeoff_reconsideration=("tradeoff_reconsideration", "mean"),
        mean_trust=("trust_mean", "mean"),
        mean_autonomy=("autonomy_mean", "mean"),
        mean_confidence=("confidence_mean", "mean"),
    ).reset_index()
    summary.to_csv(OUT, index=False)
    print("\nSaved:", OUT)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
