"""
app.py — Streamlit A/B Test Dashboard
--------------------------------------
Interactive dashboard allowing non-technical stakeholders to explore:
- Real-time conversion rate comparisons
- Confidence intervals
- Power analysis curves
- Auto-generated business recommendation memo

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import chi2_contingency, norm

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="A/B Test Dashboard | Veer Bajpai",
    page_icon="🧪",
    layout="wide",
)

ALPHA = 0.05

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def simulate(n: int, cvr_a: float, cvr_b: float, rev_a: float, rev_b: float, seed: int = 42):
    np.random.seed(seed)
    n_a, n_b = n // 2, n - n // 2
    conv_a   = np.random.binomial(1, cvr_a, n_a)
    conv_b   = np.random.binomial(1, cvr_b, n_b)
    rev_ua   = np.where(conv_a == 1, np.random.lognormal(np.log(rev_a), 0.4, n_a), 0)
    rev_ub   = np.where(conv_b == 1, np.random.lognormal(np.log(rev_b), 0.4, n_b), 0)
    df_a     = pd.DataFrame({"variant": "A", "converted": conv_a, "revenue": rev_ua})
    df_b     = pd.DataFrame({"variant": "B", "converted": conv_b, "revenue": rev_ub})
    return pd.concat([df_a, df_b], ignore_index=True)


def run_tests(df):
    a, b     = df[df["variant"] == "A"], df[df["variant"] == "B"]
    ct       = pd.crosstab(df["variant"], df["converted"])
    chi2, p_chi, _, _ = chi2_contingency(ct)
    cvr_a    = a["converted"].mean() * 100
    cvr_b    = b["converted"].mean() * 100

    t_stat, p_t = stats.ttest_ind(a["revenue"], b["revenue"], equal_var=False)
    diff     = b["revenue"].mean() - a["revenue"].mean()
    se       = np.sqrt(a["revenue"].var() / len(a) + b["revenue"].var() / len(b))
    ci       = (diff - norm.ppf(0.975) * se, diff + norm.ppf(0.975) * se)

    return {
        "cvr_a": cvr_a, "cvr_b": cvr_b,
        "lift":  (cvr_b - cvr_a) / cvr_a * 100,
        "chi2": chi2, "p_chi": p_chi, "sig_chi": p_chi < ALPHA,
        "rev_a": a["revenue"].mean(), "rev_b": b["revenue"].mean(),
        "t_stat": t_stat, "p_t": p_t, "ci": ci, "sig_t": p_t < ALPHA,
    }


# ── Sidebar Controls ──────────────────────────────────────────────────────────
st.sidebar.title("🔧 Experiment Controls")
st.sidebar.markdown("Adjust parameters to explore how they affect test results.")

n_users   = st.sidebar.slider("Total Users",       1_000, 20_000, 10_000, 500)
cvr_a     = st.sidebar.slider("Variant A CVR (%)", 1.0,  30.0, 12.0, 0.5) / 100
cvr_b     = st.sidebar.slider("Variant B CVR (%)", 1.0,  30.0, 14.0, 0.5) / 100
rev_a     = st.sidebar.number_input("Avg Revenue/Converter A ($)", 5.0, 100.0, 15.20)
rev_b     = st.sidebar.number_input("Avg Revenue/Converter B ($)", 5.0, 100.0, 18.40)
monthly_t = st.sidebar.number_input("Monthly Traffic (for impact calc)", 10_000, 500_000, 50_000, 5_000)

st.sidebar.markdown("---")
st.sidebar.markdown("**Author:** Veer Bajpai")
st.sidebar.markdown("[LinkedIn](https://linkedin.com/in/veer-bajpai) • [GitHub](https://github.com/veer-bajpai)")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🧪 A/B Test Simulation Dashboard")
st.markdown("Interactive analysis of a two-variant landing page experiment with statistical significance testing.")

df  = simulate(n_users, cvr_a, cvr_b, rev_a, rev_b)
res = run_tests(df)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Variant A CVR",  f"{res['cvr_a']:.2f}%")
col2.metric("Variant B CVR",  f"{res['cvr_b']:.2f}%", f"{res['lift']:+.1f}% lift")
col3.metric("Chi² p-value",   f"{res['p_chi']:.4f}", "✅ Significant" if res["sig_chi"] else "❌ Not Significant")
col4.metric("T-Test p-value", f"{res['p_t']:.4f}",   "✅ Significant" if res["sig_t"]  else "❌ Not Significant")

st.divider()

# ── Charts Row 1 ──────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.subheader("📊 Conversion Rate Comparison")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(["Variant A\n(Control)", "Variant B\n(Treatment)"],
                  [res["cvr_a"], res["cvr_b"]],
                  color=["#4C72B0", "#55A868"], width=0.4, alpha=0.88)
    for bar, val in zip(bars, [res["cvr_a"], res["cvr_b"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Conversion Rate (%)")
    ax.set_ylim(0, max(res["cvr_a"], res["cvr_b"]) * 1.3)
    sig_label = "p < 0.05 ✅" if res["sig_chi"] else "p ≥ 0.05 ❌"
    ax.set_title(f"Chi-Square Test: {sig_label}")
    st.pyplot(fig)
    plt.close()

with c2:
    st.subheader("💰 Revenue per User (95% CI)")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    means = [res["rev_a"], res["rev_b"]]
    ci_half = (res["ci"][1] - res["ci"][0]) / 2
    err = [0.3, ci_half]
    ax.errorbar(["Variant A", "Variant B"], means, yerr=err,
                fmt="o", capsize=10, markersize=12,
                color=["#4C72B0", "#55A868"][0], ecolor="grey")
    for i, (x, m) in enumerate(zip(["Variant A", "Variant B"], means)):
        ax.text(i, m + ci_half + 0.2, f"${m:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Avg Revenue per User ($)")
    ax.set_title("Welch T-Test: Revenue Difference")
    st.pyplot(fig)
    plt.close()

# ── Charts Row 2 ──────────────────────────────────────────────────────────────
c3, c4 = st.columns(2)

with c3:
    st.subheader("📈 Revenue Distribution (Converters)")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for variant, color in [("A", "#4C72B0"), ("B", "#55A868")]:
        sub = df[(df["variant"] == variant) & (df["revenue"] > 0)]["revenue"]
        sub.plot.kde(ax=ax, label=f"Variant {variant} (μ=${sub.mean():.2f})", color=color, linewidth=2)
    ax.set_xlabel("Revenue ($)")
    ax.legend()
    st.pyplot(fig)
    plt.close()

with c4:
    st.subheader("⚡ Statistical Power Curve")
    try:
        from statsmodels.stats.proportion import proportion_effectsize, zt_ind_solve_power
        sizes   = np.arange(200, 15_000, 200)
        es      = proportion_effectsize(cvr_a, cvr_b)
        powers  = [zt_ind_solve_power(effect_size=es, nobs1=n, alpha=ALPHA, alternative="two-sided") * 100
                   for n in sizes]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(sizes, powers, color="#2196F3", linewidth=2)
        ax.axhline(80, color="orange", linestyle="--", label="80% threshold")
        ax.axhline(90, color="red",    linestyle="--", label="90% threshold")
        ax.axvline(n_users // 2, color="green", linestyle="-.", label=f"Current n={n_users//2:,}")
        ax.set_xlabel("Sample Size per Variant")
        ax.set_ylabel("Power (%)")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close()
    except ImportError:
        st.info("Install `statsmodels` to see the power curve.")

# ── Business Recommendation ───────────────────────────────────────────────────
st.divider()
st.subheader("📝 Auto-Generated Business Recommendation Memo")

uplift = monthly_t * (res["cvr_b"] / 100) * rev_b - monthly_t * (res["cvr_a"] / 100) * rev_a
decision_color = "green" if (res["sig_chi"] and res["sig_t"]) else "red"
decision_text  = "✅ ADOPT Variant B" if (res["sig_chi"] and res["sig_t"]) else "❌ Results are NOT statistically significant"

st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| **Decision** | **:{decision_color}[{decision_text}]** |
| Variant A Conversion Rate | {res['cvr_a']:.2f}% |
| Variant B Conversion Rate | {res['cvr_b']:.2f}% |
| Relative Lift | {res['lift']:.1f}% |
| Chi² p-value | {res['p_chi']:.5f} {'✅' if res['sig_chi'] else '❌'} |
| T-test p-value | {res['p_t']:.5f} {'✅' if res['sig_t'] else '❌'} |
| Revenue/user (A) | ${res['rev_a']:.2f} |
| Revenue/user (B) | ${res['rev_b']:.2f} |
| Projected Monthly Uplift | **${uplift:,.0f}** |
""")

if res["sig_chi"] and res["sig_t"]:
    st.success(
        f"Variant B delivers a statistically significant **{res['lift']:.1f}% conversion rate lift** (p={res['p_chi']:.4f} < 0.05) "
        f"and **${res['rev_b'] - res['rev_a']:.2f} higher revenue per user**. "
        f"Rolling out to 100% of traffic is projected to generate **${uplift:,.0f} incremental monthly revenue** "
        f"based on {monthly_t:,} monthly visitors."
    )
else:
    st.warning("Results are not statistically significant at the 95% confidence level. Consider increasing sample size or running the test longer.")
