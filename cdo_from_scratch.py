import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Force the full table to print
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# -----------------------------
# Project Parameters
# -----------------------------

# Simulation Parameters
N_SIM = 1000
N_BONDS = 10

# Timing Parameters
YEARS = 5
QUARTERS_PER_YEAR = 4
N_QUARTERS = YEARS * QUARTERS_PER_YEAR

# Bond collateral characteristics
FACE_VALUE = 10_000_000
COUPON_RATE = 0.06

# Risk parameters
ANNUAL_DEFAULT_PROB = 0.04
LGD = 0.60
RECOVERY_RATE = 1 - LGD

# Correlation models how likely bonds are to default together
DEFAULT_CORRELATION = 0.20

# Part II Step 1 Parameters:
# Class A ($15 MM principal, target rate 4.5%)
CLASS_A_NOTIONAL = 15_000_000
CLASS_A_COUPON = 0.045

# Class B ($30 MM principal, target rate 5.5%)
CLASS_B_NOTIONAL = 30_000_000
CLASS_B_COUPON = 0.055

# Class C (Equity / Residual Tranche gets whatever is left over)
EQUITY_NOTIONAL = (N_BONDS * FACE_VALUE) - CLASS_A_NOTIONAL - CLASS_B_NOTIONAL

# Risk-free interest rate (4% p.a.)
RISK_FREE_RATE = 0.04

# For Reproducibility of random numbers
SEED = 123

# Pre-calculate quarterly coupon payments
quarterly_bond_coupon = FACE_VALUE * COUPON_RATE / QUARTERS_PER_YEAR

quarterly_A_coupon = (
    CLASS_A_NOTIONAL * CLASS_A_COUPON / QUARTERS_PER_YEAR
)

quarterly_B_coupon = (
    CLASS_B_NOTIONAL * CLASS_B_COUPON / QUARTERS_PER_YEAR
)

# print("Bond quarterly coupon:", quarterly_bond_coupon)
# print("Class A quarterly coupon:", quarterly_A_coupon)
# print("Class B quarterly coupon:", quarterly_B_coupon)

# -----------------------------
# Step 2: Correlation Matrix
# -----------------------------

# Create a 10x10 matrix where every pair of bonds has a 0.20 correlation
corr_matrix = np.full(
    (N_BONDS, N_BONDS),
    DEFAULT_CORRELATION
)

# A bond's correlation with itself is exactly 1.0
np.fill_diagonal(corr_matrix, 1.0)

# print("\nCorrelation matrix:")
# print(corr_matrix)

# Do Cholesky decomposition
cholesky_matrix = np.linalg.cholesky(corr_matrix)

# print("\nCholesky matrix:")
# print(cholesky_matrix)

# -----------------------------
# Step 3: Generate Random Normals (with Moment Matching for Part II Step 2)
# -----------------------------

np.random.seed(SEED)

# Generate 1,000 sets of 10 independent random numbers from a standard normal distribution
independent_normals_raw = np.random.normal(
    size=(N_SIM, N_BONDS)
)

# Part II Step 2: Match first and second moments (mean = 0, std = 1) of static random numbers
independent_normals = (
    independent_normals_raw - independent_normals_raw.mean(axis=0)
) / independent_normals_raw.std(axis=0, ddof=1)

# print("\nIndependent normals shape:")
# print(independent_normals.shape)

# Multiply the independent normals by the Cholesky matrix to get correlated matrix
correlated_normals = independent_normals @ cholesky_matrix.T

# Re-standardize correlated normals so each bond column has exact sample mean 0 and std 1
correlated_normals = (
    correlated_normals - correlated_normals.mean(axis=0)
) / correlated_normals.std(axis=0, ddof=1)

# print("\nCorrelated normals shape:")
# print(correlated_normals.shape)

# Verify that our simulated correlation matches our 20% target
sample_corr = np.corrcoef(
    correlated_normals,
    rowvar=False
)

# print("\nSample correlation matrix:")
# print(np.round(sample_corr, 2))

# -----------------------------
# Step 4: Convert Normal to Uniform
# -----------------------------

# Convert correlated normals to uniform probabilities from 0 to 1
uniforms = norm.cdf(correlated_normals)

# print("\nUniforms shape:")
# print(uniforms.shape)

# print("\nFirst simulation - uniform values:")
# print(uniforms[0])

# -----------------------------
# Step 5: Convert Uniforms to Default Times
# -----------------------------

# Use the inverse of the exponential distribution CDF to translate the
# uniform probabilities [0, 1] into an actual time-to-default in years.
default_times = (
    np.log(1 - uniforms)
    / np.log(1 - ANNUAL_DEFAULT_PROB)
)

# print("\nDefault times shape:")
# print(default_times.shape)

# print("\nFirst simulation - default times in years:")
# print(np.round(default_times[0], 2))

# -----------------------------
# Step 6: Convert Default Times to Default Quarters
# -----------------------------

# Multiply years by 4 and round up to find the exact quarter a bond defaults.
default_quarters = np.ceil(
    default_times * QUARTERS_PER_YEAR
).astype(int)

# print("\nFirst simulation - default quarters:")
# print(default_quarters[0])

# -----------------------------
# Step 7: Build Bond Cash Flows
# -----------------------------
recovery_amount = FACE_VALUE * RECOVERY_RATE

# 3D arrays to store cash flows: [Simulation Scenario, Bond Number, Quarter]
coupon_cash_flows = np.zeros(
    (N_SIM, N_BONDS, N_QUARTERS)
)

recovery_cash_flows = np.zeros(
    (N_SIM, N_BONDS, N_QUARTERS)
)

principal_cash_flows = np.zeros(
    (N_SIM, N_BONDS, N_QUARTERS)
)

for sim in range(N_SIM):

    for bond in range(N_BONDS):
        # If the bond defaults
        default_q = default_quarters[sim, bond]

        for q in range(1, N_QUARTERS + 1):

            # Bond defaults within the 5-year horizon
            if default_q <= N_QUARTERS:

                if q < default_q:
                    coupon_cash_flows[sim, bond, q - 1] = quarterly_bond_coupon

                elif q == default_q:
                    recovery_cash_flows[sim, bond, q - 1] = recovery_amount

            # Bond survives to maturity
            else:

                if q <= N_QUARTERS:
                    coupon_cash_flows[sim, bond, q - 1] = quarterly_bond_coupon

                if q == N_QUARTERS:
                    principal_cash_flows[sim, bond, q - 1] = FACE_VALUE

# Aggregate the cash flows from all 10 bonds for the whole pool
collateral_interest_cf = coupon_cash_flows.sum(axis=1)

collateral_recovery_cf = recovery_cash_flows.sum(axis=1)

collateral_principal_cf = principal_cash_flows.sum(axis=1)

# This is the total cash coming in to the CDO every quarter
total_collateral_cf = (
    collateral_interest_cf
    + collateral_recovery_cf
    + collateral_principal_cf
)

# print("\nCase 1 - Interest CF:")
# print(collateral_interest_cf[0])

# print("\nCase 1 - Recovery CF:")
# print(collateral_recovery_cf[0])

# print("\nCase 1 - Principal CF:")
# print(collateral_principal_cf[0])

# print("\nCase 1 - Total CF:")
# print(total_collateral_cf[0])

# -----------------------------
# Step 8: CDO Waterfall
# -----------------------------

# Initialize arrays to track what gets paid out to the tranches
class_A_interest_cf = np.zeros((N_SIM, N_QUARTERS))
class_B_interest_cf = np.zeros((N_SIM, N_QUARTERS))

class_A_principal_cf = np.zeros((N_SIM, N_QUARTERS))
class_B_principal_cf = np.zeros((N_SIM, N_QUARTERS))

equity_cf = np.zeros((N_SIM, N_QUARTERS))

# Track the remaining principal balance owed to Class A and B
class_A_balance = np.zeros((N_SIM, N_QUARTERS + 1))
class_B_balance = np.zeros((N_SIM, N_QUARTERS + 1))

# Set initial balances at Quarter 0
class_A_balance[:, 0] = CLASS_A_NOTIONAL
class_B_balance[:, 0] = CLASS_B_NOTIONAL

for sim in range(N_SIM):

    for q in range(N_QUARTERS):

        # Beginning-of-quarter balances
        A_bal = class_A_balance[sim, q]
        B_bal = class_B_balance[sim, q]

        # ---------------------------------
        # 1. Interest Waterfall for A and B
        # ---------------------------------

        available_interest = collateral_interest_cf[sim, q]

        A_interest_due = A_bal * CLASS_A_COUPON / QUARTERS_PER_YEAR
        B_interest_due = B_bal * CLASS_B_COUPON / QUARTERS_PER_YEAR

        A_interest_paid = min(
            available_interest,
            A_interest_due
        )

        remaining_interest = available_interest - A_interest_paid

        B_interest_paid = min(
            remaining_interest,
            B_interest_due
        )

        remaining_interest -= B_interest_paid

        # ----------------------------------
        # 2. Principal Waterfall for A and B
        # ----------------------------------

        available_principal = (
            collateral_recovery_cf[sim, q]
            + collateral_principal_cf[sim, q]
        )

        A_principal_paid = min(
            available_principal,
            A_bal
        )

        remaining_principal = (
            available_principal
            - A_principal_paid
        )

        B_principal_paid = min(
            remaining_principal,
            B_bal
        )

        remaining_principal -= B_principal_paid

        # -------------------------
        # 3. Equity Gets Residual
        # -------------------------

        equity_paid = (
            remaining_interest
            + remaining_principal
        )

        # Save cash flows
        class_A_interest_cf[sim, q] = A_interest_paid
        class_B_interest_cf[sim, q] = B_interest_paid

        class_A_principal_cf[sim, q] = A_principal_paid
        class_B_principal_cf[sim, q] = B_principal_paid

        equity_cf[sim, q] = equity_paid

        # Update balances
        class_A_balance[sim, q + 1] = (
            A_bal - A_principal_paid
        )

        class_B_balance[sim, q + 1] = (
            B_bal - B_principal_paid
        )

# Combine interest and principal for total tranche cash flows
class_A_total_cf = (
    class_A_interest_cf
    + class_A_principal_cf
)

class_B_total_cf = (
    class_B_interest_cf
    + class_B_principal_cf
)

# print("\nCase 1 - Class A Total CF:")
# print(class_A_total_cf[0])

# print("\nCase 1 - Class B Total CF:")
# print(class_B_total_cf[0])

# print("\nCase 1 - Equity CF:")
# print(equity_cf[0])

# print("\nCase 1 - Class A Balance:")
# print(class_A_balance[0])

# print("\nCase 1 - Class B Balance:")
# print(class_B_balance[0])

allocated_cf = (
    class_A_total_cf
    + class_B_total_cf
    + equity_cf
)

# print(
#     np.allclose(
#         allocated_cf,
#         total_collateral_cf
#     )
# )

bond_cash_flows = (
    coupon_cash_flows
    + recovery_cash_flows
    + principal_cash_flows
)

def view_case(case_num):
    """
    Display detailed cash flows for one simulation case.
    case_num: 1 to 1000
    """

    if case_num < 1 or case_num > N_SIM:
        raise ValueError("Case number must be between 1 and 1000.")

    idx = case_num - 1

    # -----------------------------
    # 1. Individual Bond Cash Flows
    # -----------------------------

    bond_df = pd.DataFrame(
        bond_cash_flows[idx].T,
        index=[f"Q{i}" for i in range(1, N_QUARTERS + 1)],
        columns=[f"Bond {i}" for i in range(1, N_BONDS + 1)]
    )

    print("\n" + "=" * 100)
    print(f"CASE {case_num}: INDIVIDUAL BOND CASH FLOWS")
    print("=" * 100)

    print(
        bond_df.map(
            lambda x: f"{x:,.0f}"
        )
    )

    # -----------------------------
    # 2. Collateral + CDO Waterfall
    # -----------------------------

    waterfall_df = pd.DataFrame(
        {
            "Collateral Interest": collateral_interest_cf[idx],
            "Recovery": collateral_recovery_cf[idx],
            "Principal": collateral_principal_cf[idx],
            "Total Collateral": total_collateral_cf[idx],
            "Class A": class_A_total_cf[idx],
            "Class B": class_B_total_cf[idx],
            "Equity": equity_cf[idx],
        },
        index=[f"Q{i}" for i in range(1, N_QUARTERS + 1)]
    )

    print("\n" + "=" * 100)
    print(f"CASE {case_num}: CDO WATERFALL")
    print("=" * 100)

    print(
        waterfall_df.map(
            lambda x: f"{x:,.0f}"
        )
    )

# -----------------------------
# Step 9: Statistical Analysis
# -----------------------------

total_cf_collateral = total_collateral_cf.sum(axis=1)
total_cf_A = class_A_total_cf.sum(axis=1)
total_cf_B = class_B_total_cf.sum(axis=1)
total_cf_equity = equity_cf.sum(axis=1)

results_df = pd.DataFrame({
    "Collateral": total_cf_collateral,
    "Class A": total_cf_A,
    "Class B": total_cf_B,
    "Equity": total_cf_equity
})

stats = results_df.describe(
    percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
).T

print("\n" + "=" * 100)
print("TASK 3: TOTAL 5-YEAR CASH FLOW STATISTICS ($MM)")
print("=" * 100)

stats_display = stats.copy()

for col in stats_display.columns:
    if col != "count":
        stats_display[col] = stats_display[col] / 1_000_000

print(stats_display.round(2))

# -----------------------------
# True Payment Shortfall Analysis
# -----------------------------

# Interest due each quarter based on beginning balance
A_interest_due = (
    class_A_balance[:, :-1]
    * CLASS_A_COUPON
    / QUARTERS_PER_YEAR
)

B_interest_due = (
    class_B_balance[:, :-1]
    * CLASS_B_COUPON
    / QUARTERS_PER_YEAR
)

# Unpaid interest
A_interest_shortfall = np.maximum(
    A_interest_due - class_A_interest_cf,
    0
)

B_interest_shortfall = np.maximum(
    B_interest_due - class_B_interest_cf,
    0
)

# Probability that any interest payment is missed
prob_A_interest_shortfall = np.mean(
    A_interest_shortfall.sum(axis=1) > 1e-6
)

prob_B_interest_shortfall = np.mean(
    B_interest_shortfall.sum(axis=1) > 1e-6
)

# Principal still unpaid after Q20
prob_A_principal_shortfall = np.mean(
    class_A_balance[:, -1] > 1e-6
)

prob_B_principal_shortfall = np.mean(
    class_B_balance[:, -1] > 1e-6
)

print("\nTrue Payment Shortfall Probabilities")
print("------------------------------------")
print(
    f"Class A interest shortfall: "
    f"{prob_A_interest_shortfall:.2%}"
)
print(
    f"Class B interest shortfall: "
    f"{prob_B_interest_shortfall:.2%}"
)
print(
    f"Class A principal shortfall: "
    f"{prob_A_principal_shortfall:.2%}"
)
print(
    f"Class B principal shortfall: "
    f"{prob_B_principal_shortfall:.2%}"
)

# -----------------------------
# Step 10: Probability Distributions
# -----------------------------
os.makedirs("distributions", exist_ok=True)

plt.figure()

plt.hist(
    total_cf_collateral / 1_000_000,
    bins=30
)

plt.xlabel("Total 5-Year Cash Flow ($MM)")
plt.ylabel("Frequency")
plt.title("Distribution of Collateral Cash Flows")
plt.savefig("distributions/collateral_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure()

plt.hist(
    total_cf_A / 1_000_000,
    bins=30
)

plt.xlabel("Total 5-Year Cash Flow ($MM)")
plt.ylabel("Frequency")
plt.title("Distribution of Class A Cash Flows")
plt.savefig("distributions/class_A_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure()

plt.hist(
    total_cf_B / 1_000_000,
    bins=30
)

plt.xlabel("Total 5-Year Cash Flow ($MM)")
plt.ylabel("Frequency")
plt.title("Distribution of Class B Cash Flows")
plt.savefig("distributions/class_B_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure()

plt.hist(
    total_cf_equity / 1_000_000,
    bins=30
)

plt.xlabel("Total 5-Year Cash Flow ($MM)")
plt.ylabel("Frequency")
plt.title("Distribution of Equity Cash Flows")
plt.savefig("distributions/equity_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# Step 11: Sensitivity Analysis
# -----------------------------

def run_scenario(pd_annual, lgd, rho, seed=123):

    np.random.seed(seed)

    recovery_rate = 1 - lgd
    recovery_amount = FACE_VALUE * recovery_rate

    # Correlation matrix
    corr_matrix = np.full((N_BONDS, N_BONDS), rho)
    np.fill_diagonal(corr_matrix, 1.0)

    L = np.linalg.cholesky(corr_matrix)

    # Independent normals -> moment-matched -> correlated normals
    independent_normals_s = np.random.normal(
        size=(N_SIM, N_BONDS)
    )
    independent_normals_s = (
        independent_normals_s - independent_normals_s.mean(axis=0)
    ) / independent_normals_s.std(axis=0, ddof=1)

    correlated_normals_s = independent_normals_s @ L.T
    correlated_normals_s = (
        correlated_normals_s - correlated_normals_s.mean(axis=0)
    ) / correlated_normals_s.std(axis=0, ddof=1)

    # Normal -> Uniform
    uniforms_s = norm.cdf(correlated_normals_s)

    # Uniform -> default time
    default_times_s = (
        np.log(1 - uniforms_s)
        / np.log(1 - pd_annual)
    )

    default_quarters_s = np.ceil(
        default_times_s * QUARTERS_PER_YEAR
    ).astype(int)

    # Bond cash-flow components
    coupon_cf = np.zeros(
        (N_SIM, N_BONDS, N_QUARTERS)
    )

    recovery_cf = np.zeros(
        (N_SIM, N_BONDS, N_QUARTERS)
    )

    principal_cf = np.zeros(
        (N_SIM, N_BONDS, N_QUARTERS)
    )

    for sim in range(N_SIM):
        for bond in range(N_BONDS):

            dq = default_quarters_s[sim, bond]

            for q in range(1, N_QUARTERS + 1):

                if dq <= N_QUARTERS:

                    if q < dq:
                        coupon_cf[sim, bond, q - 1] = quarterly_bond_coupon

                    elif q == dq:
                        recovery_cf[sim, bond, q - 1] = recovery_amount

                else:

                    coupon_cf[sim, bond, q - 1] = quarterly_bond_coupon

                    if q == N_QUARTERS:
                        principal_cf[sim, bond, q - 1] = FACE_VALUE

    # Aggregate collateral cash flows
    collateral_interest = coupon_cf.sum(axis=1)
    collateral_recovery = recovery_cf.sum(axis=1)
    collateral_principal = principal_cf.sum(axis=1)

    collateral_total = (
        collateral_interest
        + collateral_recovery
        + collateral_principal
    )

    # Waterfall arrays
    class_A_interest_cf_s = np.zeros((N_SIM, N_QUARTERS))
    class_B_interest_cf_s = np.zeros((N_SIM, N_QUARTERS))

    class_A_principal_cf_s = np.zeros((N_SIM, N_QUARTERS))
    class_B_principal_cf_s = np.zeros((N_SIM, N_QUARTERS))

    equity_cf_s = np.zeros((N_SIM, N_QUARTERS))

    class_A_balance_s = np.zeros((N_SIM, N_QUARTERS + 1))
    class_B_balance_s = np.zeros((N_SIM, N_QUARTERS + 1))

    class_A_balance_s[:, 0] = CLASS_A_NOTIONAL
    class_B_balance_s[:, 0] = CLASS_B_NOTIONAL

    # Waterfall
    for sim in range(N_SIM):

        for q in range(N_QUARTERS):

            A_bal = class_A_balance_s[sim, q]
            B_bal = class_B_balance_s[sim, q]

            # Interest waterfall
            available_interest = collateral_interest[sim, q]

            A_interest_due = (
                A_bal * CLASS_A_COUPON / QUARTERS_PER_YEAR
            )

            B_interest_due = (
                B_bal * CLASS_B_COUPON / QUARTERS_PER_YEAR
            )

            A_interest_paid = min(
                available_interest,
                A_interest_due
            )

            remaining_interest = (
                available_interest
                - A_interest_paid
            )

            B_interest_paid = min(
                remaining_interest,
                B_interest_due
            )

            remaining_interest -= B_interest_paid

            # Principal waterfall
            available_principal = (
                collateral_recovery[sim, q]
                + collateral_principal[sim, q]
            )

            A_principal_paid = min(
                available_principal,
                A_bal
            )

            remaining_principal = (
                available_principal
                - A_principal_paid
            )

            B_principal_paid = min(
                remaining_principal,
                B_bal
            )

            remaining_principal -= B_principal_paid

            # Equity residual
            equity_paid = (
                remaining_interest
                + remaining_principal
            )

            class_A_interest_cf_s[sim, q] = A_interest_paid
            class_B_interest_cf_s[sim, q] = B_interest_paid

            class_A_principal_cf_s[sim, q] = A_principal_paid
            class_B_principal_cf_s[sim, q] = B_principal_paid

            equity_cf_s[sim, q] = equity_paid

            class_A_balance_s[sim, q + 1] = (
                A_bal - A_principal_paid
            )

            class_B_balance_s[sim, q + 1] = (
                B_bal - B_principal_paid
            )

    class_A_total_cf_s = (
        class_A_interest_cf_s
        + class_A_principal_cf_s
    )

    class_B_total_cf_s = (
        class_B_interest_cf_s
        + class_B_principal_cf_s
    )

    total_collateral_s = collateral_total.sum(axis=1)
    total_A_s = class_A_total_cf_s.sum(axis=1)
    total_B_s = class_B_total_cf_s.sum(axis=1)
    total_equity_s = equity_cf_s.sum(axis=1)

    return {
        "Collateral": total_collateral_s,
        "Class A": total_A_s,
        "Class B": total_B_s,
        "Equity": total_equity_s
    }


scenarios = {
    "Base": (0.04, 0.60, 0.20),
    "Low PD": (0.02, 0.60, 0.20),
    "High PD": (0.06, 0.60, 0.20),
    "Low LGD": (0.04, 0.40, 0.20),
    "High LGD": (0.04, 0.80, 0.20),
    "Low Corr": (0.04, 0.60, 0.10),
    "High Corr": (0.04, 0.60, 0.40),
}

sensitivity_rows = []

for name, (pd_val, lgd_val, rho_val) in scenarios.items():

    result = run_scenario(
        pd_annual=pd_val,
        lgd=lgd_val,
        rho=rho_val
    )

    sensitivity_rows.append({
        "Scenario": name,

        "Collateral Mean": result["Collateral"].mean() / 1_000_000,
        "Collateral Std": result["Collateral"].std() / 1_000_000,
        "Collateral 5%": np.percentile(result["Collateral"], 5) / 1_000_000,

        "Class A Mean": result["Class A"].mean() / 1_000_000,

        "Class B Mean": result["Class B"].mean() / 1_000_000,

        "Equity Mean": result["Equity"].mean() / 1_000_000,
        "Equity Std": result["Equity"].std() / 1_000_000,
        "Equity 5%": np.percentile(result["Equity"], 5) / 1_000_000,
    })

sensitivity_df = pd.DataFrame(sensitivity_rows)

print("\n" + "=" * 100)
print("SENSITIVITY ANALYSIS ($MM)")
print("=" * 100)

print(sensitivity_df.round(2))

os.makedirs("sensitivity", exist_ok=True)
sensitivity_df.to_csv(
    "sensitivity/sensitivity_analysis.csv",
    index=False
)

# -----------------------------
# Step 12: Sensitivity Charts
# -----------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    sensitivity_df["Scenario"],
    sensitivity_df["Collateral 5%"]
)

plt.ylabel("Collateral 5th Percentile ($MM)")
plt.title("Sensitivity of Collateral Tail Cash Flow")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "sensitivity/sensitivity_collateral_5pct.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

plt.figure(figsize=(10, 6))

plt.bar(
    sensitivity_df["Scenario"],
    sensitivity_df["Equity 5%"]
)

plt.ylabel("Equity 5th Percentile ($MM)")
plt.title("Sensitivity of Equity Tail Cash Flow")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "sensitivity/sensitivity_equity_5pct.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==============================================================================
# PART II: CDO VALUATION USING THE LINTNER MODEL
# ==============================================================================

# ------------------------------------------------------------------------------
# Step 13 (Part II Steps 1-3): Scheduled Waterfall, Risk-Free PVs, Covariance
# Matrix of Tranche Cash Flows, and Scenario Default Indicators
# ------------------------------------------------------------------------------

quarters_idx = np.arange(1, N_QUARTERS + 1)
quarterly_rf = RISK_FREE_RATE / QUARTERS_PER_YEAR
rf_discount_factors = 1.0 / ((1.0 + quarterly_rf) ** quarters_idx)

# 1. Scheduled (Zero-Default) Cash Flows for Collateral and Tranches
scheduled_collateral_cf = np.full(N_QUARTERS, N_BONDS * quarterly_bond_coupon)
scheduled_collateral_cf[-1] += N_BONDS * FACE_VALUE

scheduled_A_cf = np.full(N_QUARTERS, quarterly_A_coupon)
scheduled_A_cf[-1] += CLASS_A_NOTIONAL

scheduled_B_cf = np.full(N_QUARTERS, quarterly_B_coupon)
scheduled_B_cf[-1] += CLASS_B_NOTIONAL

scheduled_equity_cf = scheduled_collateral_cf - scheduled_A_cf - scheduled_B_cf

scheduled_waterfall_df = pd.DataFrame(
    {
        "Scheduled Collateral ($MM)": scheduled_collateral_cf / 1_000_000,
        "Scheduled Class A ($MM)": scheduled_A_cf / 1_000_000,
        "Scheduled Class B ($MM)": scheduled_B_cf / 1_000_000,
        "Scheduled Residual ($MM)": scheduled_equity_cf / 1_000_000,
        "RF Discount Factor": rf_discount_factors,
    },
    index=[f"Q{i}" for i in range(1, N_QUARTERS + 1)],
)

print("\n" + "=" * 100)
print("PART II - STEP 1: SCHEDULED WATERFALL CASH FLOWS ($MM)")
print("=" * 100)
print(scheduled_waterfall_df.round(4))

# 2. Verify Moment Matching on the 1,000 Static Random Number Simulations
print("\n" + "=" * 100)
print("PART II - STEP 2: MOMENT-MATCHED STATIC NORMAL VERIFICATION (1,000 SIMS)")
print("=" * 100)
moment_check_df = pd.DataFrame(
    {
        "Raw Mean": independent_normals_raw.mean(axis=0),
        "Raw Std": independent_normals_raw.std(axis=0, ddof=1),
        "Moment-Matched Mean": independent_normals.mean(axis=0),
        "Moment-Matched Std": independent_normals.std(axis=0, ddof=1),
        "Correlated Normal Mean": correlated_normals.mean(axis=0),
        "Correlated Normal Std": correlated_normals.std(axis=0, ddof=1),
    },
    index=[f"Bond {i}" for i in range(1, N_BONDS + 1)],
)
print(moment_check_df.round(6))

# 3. Compute PV of each tranche in each scenario at the risk-free rate (4% p.a.)
pv_collateral_rf = total_collateral_cf @ rf_discount_factors
pv_A_rf = class_A_total_cf @ rf_discount_factors
pv_B_rf = class_B_total_cf @ rf_discount_factors
pv_equity_rf = equity_cf @ rf_discount_factors

# Default indicators per scenario:
# A tranche defaults in scenario s if it has any unpaid interest or unpaid ending principal
# (equivalently, its realized PV discounted at its target rate is strictly below par).
default_indicator_A = (
    (A_interest_shortfall.sum(axis=1) > 1e-6)
    | (class_A_balance[:, -1] > 1e-6)
).astype(int)

default_indicator_B = (
    (B_interest_shortfall.sum(axis=1) > 1e-6)
    | (class_B_balance[:, -1] > 1e-6)
).astype(int)

default_indicator_any = np.maximum(default_indicator_A, default_indicator_B)

# 3x3 Covariance Matrix of the three tranches' risk-free PVs
tranche_pv_matrix = np.vstack([pv_A_rf, pv_B_rf, pv_equity_rf])
cov_matrix_tranches = np.cov(tranche_pv_matrix, ddof=1)
corr_matrix_tranches = np.corrcoef(tranche_pv_matrix)

cov_df = pd.DataFrame(
    cov_matrix_tranches / 1e12,
    index=["Class A", "Class B", "Residual (Equity)"],
    columns=["Class A", "Class B", "Residual (Equity)"],
)

corr_df = pd.DataFrame(
    corr_matrix_tranches,
    index=["Class A", "Class B", "Residual (Equity)"],
    columns=["Class A", "Class B", "Residual (Equity)"],
)

print("\n" + "=" * 100)
print("PART II - STEP 3: RISK-FREE PVs, COVARIANCE MATRIX ($MM^2), & DEFAULT INDICATORS")
print("=" * 100)

pv_summary_df = pd.DataFrame(
    {
        "Notional ($MM)": [
            CLASS_A_NOTIONAL / 1e6,
            CLASS_B_NOTIONAL / 1e6,
            EQUITY_NOTIONAL / 1e6,
            (N_BONDS * FACE_VALUE) / 1e6,
        ],
        "Scheduled PV at RF ($MM)": [
            (scheduled_A_cf @ rf_discount_factors) / 1e6,
            (scheduled_B_cf @ rf_discount_factors) / 1e6,
            (scheduled_equity_cf @ rf_discount_factors) / 1e6,
            (scheduled_collateral_cf @ rf_discount_factors) / 1e6,
        ],
        "Mean Simulated PV at RF ($MM)": [
            pv_A_rf.mean() / 1e6,
            pv_B_rf.mean() / 1e6,
            pv_equity_rf.mean() / 1e6,
            pv_collateral_rf.mean() / 1e6,
        ],
        "Std Dev of PV ($MM)": [
            pv_A_rf.std(ddof=1) / 1e6,
            pv_B_rf.std(ddof=1) / 1e6,
            pv_equity_rf.std(ddof=1) / 1e6,
            pv_collateral_rf.std(ddof=1) / 1e6,
        ],
        "Default Indicator Freq (%)": [
            default_indicator_A.mean() * 100,
            default_indicator_B.mean() * 100,
            np.nan,
            default_indicator_any.mean() * 100,
        ],
    },
    index=["Class A", "Class B", "Residual (Equity)", "Total Collateral Pool"],
)
print(pv_summary_df.round(4))

print("\nCovariance Matrix of Tranche PVs at Risk-Free Rate ($MM^2):")
print(cov_df.round(6))

print("\nCorrelation Matrix of Tranche PVs at Risk-Free Rate:")
print(corr_df.round(4))


# ------------------------------------------------------------------------------
# Step 14 (Part II Steps 4-5): Implicit Cost of Risk, Cost per Unit Risk, and
# Lintner Tranche Valuation
# ------------------------------------------------------------------------------
# Note on Bond Valuation & Parameter Assumption (per Professor's PS):
# - With the Part I collateral parameters (6.0% coupon, 4.0% annual default prob,
#   60% LGD), the expected net return (~3.6%) is below the 4.0% risk-free rate,
#   so the mean risk-free PV of the 10 bonds is $97.69 MM (< $100.00 MM par).
# - Method A (Original 6% Bond Coupon): Value the 10 bonds by discounting their
#   expected cash flows at the 6.0% risky market yield ($89.97 MM total, or
#   $8.997 MM per bond). Then Implicit Cost of Risk = $97.69 MM - $89.97 MM = $7.72 MM.
# - Method B (Adjusted 8% Bond Coupon at $100 MM Par): If the 10 bonds are priced
#   at Par ($100.00 MM) in the market, raising the collateral bond coupon from 6.0%
#   to 8.0% gives a mean risk-free PV of $105.82 MM (> $100.00 MM Par), yielding
#   a positive Implicit Cost of Risk = $105.82 MM - $100.00 MM = $5.82 MM.

quarterly_bond_yield = COUPON_RATE / QUARTERS_PER_YEAR
bond_yield_discount_factors = 1.0 / ((1.0 + quarterly_bond_yield) ** quarters_idx)
bond_market_valuation_6pct = (total_collateral_cf @ bond_yield_discount_factors).mean()

implicit_cost_of_risk = pv_collateral_rf.mean() - bond_market_valuation_6pct

# Pool variance and standard deviation from the 3x3 tranche covariance matrix
pool_variance = cov_matrix_tranches.sum()
pool_std_dev = np.sqrt(pool_variance)

# Each tranche's covariance with the total pool (row sum of the 3x3 covariance matrix)
cov_with_pool = cov_matrix_tranches.sum(axis=1)
marginal_std_risk = cov_with_pool / pool_std_dev

# Cost per unit of risk (both in variance units and standard deviation units)
cost_per_unit_variance_risk = implicit_cost_of_risk / pool_variance
cost_per_unit_std_risk = implicit_cost_of_risk / pool_std_dev

# Lintner valuation of each tranche: V_i = E[PV_i(rf)] - lambda * Risk_i
lintner_risk_discount = cost_per_unit_variance_risk * cov_with_pool
lintner_values = np.array([pv_A_rf.mean(), pv_B_rf.mean(), pv_equity_rf.mean()]) - lintner_risk_discount

print("\n" + "=" * 100)
print("PART II - STEP 4: IMPLICIT COST OF RISK & COST PER UNIT RISK")
print("=" * 100)
print(f"Expected Collateral PV at 4.0% Risk-Free Rate : ${pv_collateral_rf.mean() / 1e6:,.4f} MM")
print(f"Market Valuation of 10 Bonds (at 6.0% Yield)  : ${bond_market_valuation_6pct / 1e6:,.4f} MM")
print(f"Implicit Cost of Risk (Total Risk Discount)   : ${implicit_cost_of_risk / 1e6:,.4f} MM")
print(f"Total Pool Variance (Sum of 3x3 Cov Matrix)   : {pool_variance / 1e12:,.4f} ($MM^2)")
print(f"Total Pool Standard Deviation (Sigma_Pool)    : ${pool_std_dev / 1e6:,.4f} MM")
print(f"Cost per Unit of Std Dev Risk (Lambda_Std)    : {cost_per_unit_std_risk:,.6f} ($ per $ of Std Risk)")
print(f"Cost per Unit of Variance Risk (Lambda_Var)   : {cost_per_unit_variance_risk * 1e6:,.6f} ($ per $MM^2 of Var)")

lintner_table_df = pd.DataFrame(
    {
        "Notional ($MM)": [
            CLASS_A_NOTIONAL / 1e6,
            CLASS_B_NOTIONAL / 1e6,
            EQUITY_NOTIONAL / 1e6,
            (N_BONDS * FACE_VALUE) / 1e6,
        ],
        "Target Rate (%)": [
            CLASS_A_COUPON * 100,
            CLASS_B_COUPON * 100,
            np.nan,
            COUPON_RATE * 100,
        ],
        "E[PV] at 4% RF ($MM)": [
            pv_A_rf.mean() / 1e6,
            pv_B_rf.mean() / 1e6,
            pv_equity_rf.mean() / 1e6,
            pv_collateral_rf.mean() / 1e6,
        ],
        "Cov with Pool ($MM^2)": [
            cov_with_pool[0] / 1e12,
            cov_with_pool[1] / 1e12,
            cov_with_pool[2] / 1e12,
            pool_variance / 1e12,
        ],
        "Risk Contrib Sigma_i ($MM)": [
            marginal_std_risk[0] / 1e6,
            marginal_std_risk[1] / 1e6,
            marginal_std_risk[2] / 1e6,
            pool_std_dev / 1e6,
        ],
        "Risk Charge ($MM)": [
            lintner_risk_discount[0] / 1e6,
            lintner_risk_discount[1] / 1e6,
            lintner_risk_discount[2] / 1e6,
            implicit_cost_of_risk / 1e6,
        ],
        "Lintner Value ($MM)": [
            lintner_values[0] / 1e6,
            lintner_values[1] / 1e6,
            lintner_values[2] / 1e6,
            lintner_values.sum() / 1e6,
        ],
        "Value / Par (%)": [
            (lintner_values[0] / CLASS_A_NOTIONAL) * 100,
            (lintner_values[1] / CLASS_B_NOTIONAL) * 100,
            (lintner_values[2] / EQUITY_NOTIONAL) * 100,
            (lintner_values.sum() / (N_BONDS * FACE_VALUE)) * 100,
        ],
    },
    index=["Class A", "Class B", "Residual (Equity)", "Total CDO"],
)

print("\n" + "=" * 100)
print("PART II - STEP 5: LINTNER TRANCHE VALUATION TABLE (BASE 6.0% BOND COUPON)")
print("=" * 100)
print(lintner_table_df.round(4))


# ------------------------------------------------------------------------------
# Helper Function for Arbitrary Structure & Collateral Coupon (Used for Par
# Calibration Check and Step 6 Structure Optimization)
# ------------------------------------------------------------------------------

def evaluate_cdo_structure(
    class_a_notional=15_000_000,
    class_a_coupon=0.045,
    class_b_notional=30_000_000,
    class_b_coupon=0.055,
    bond_coupon_rate=COUPON_RATE,
    bond_valuation_mode="yield",
):
    """
    Run the waterfall and Lintner valuation for a given CDO tranche structure.
    bond_valuation_mode:
      - 'yield': values the 10 collateral bonds at their coupon discount rate (6%)
      - 'par'  : values the 10 collateral bonds at Par ($100 MM)
    """
    scale = bond_coupon_rate / COUPON_RATE
    coll_int = collateral_interest_cf * scale
    tot_coll = coll_int + collateral_recovery_cf + collateral_principal_cf

    A_int = np.zeros((N_SIM, N_QUARTERS))
    B_int = np.zeros((N_SIM, N_QUARTERS))
    A_prin = np.zeros((N_SIM, N_QUARTERS))
    B_prin = np.zeros((N_SIM, N_QUARTERS))
    eq_cf = np.zeros((N_SIM, N_QUARTERS))

    A_bal_arr = np.zeros((N_SIM, N_QUARTERS + 1))
    B_bal_arr = np.zeros((N_SIM, N_QUARTERS + 1))
    A_bal_arr[:, 0] = class_a_notional
    B_bal_arr[:, 0] = class_b_notional

    for sim in range(N_SIM):
        for q in range(N_QUARTERS):
            ab = A_bal_arr[sim, q]
            bb = B_bal_arr[sim, q]

            avail_i = coll_int[sim, q]
            ai_due = ab * class_a_coupon / QUARTERS_PER_YEAR
            bi_due = bb * class_b_coupon / QUARTERS_PER_YEAR

            ai_paid = min(avail_i, ai_due)
            rem_i = avail_i - ai_paid
            bi_paid = min(rem_i, bi_due)
            rem_i -= bi_paid

            avail_p = collateral_recovery_cf[sim, q] + collateral_principal_cf[sim, q]
            ap_paid = min(avail_p, ab)
            rem_p = avail_p - ap_paid
            bp_paid = min(rem_p, bb)
            rem_p -= bp_paid

            eq_cf[sim, q] = rem_i + rem_p
            A_int[sim, q] = ai_paid
            B_int[sim, q] = bi_paid
            A_prin[sim, q] = ap_paid
            B_prin[sim, q] = bp_paid

            A_bal_arr[sim, q + 1] = ab - ap_paid
            B_bal_arr[sim, q + 1] = bb - bp_paid

    A_cf_tot = A_int + A_prin
    B_cf_tot = B_int + B_prin

    A_due_mat = A_bal_arr[:, :-1] * class_a_coupon / QUARTERS_PER_YEAR
    B_due_mat = B_bal_arr[:, :-1] * class_b_coupon / QUARTERS_PER_YEAR

    def_A = ((A_due_mat - A_int).sum(axis=1) > 1e-6) | (A_bal_arr[:, -1] > 1e-6)
    def_B = ((B_due_mat - B_int).sum(axis=1) > 1e-6) | (B_bal_arr[:, -1] > 1e-6)
    def_any = def_A | def_B

    pv_coll = tot_coll @ rf_discount_factors
    pv_A = A_cf_tot @ rf_discount_factors
    pv_B = B_cf_tot @ rf_discount_factors
    pv_E = eq_cf @ rf_discount_factors

    if bond_valuation_mode == "par":
        v_pool = float(N_BONDS * FACE_VALUE)
    else:
        df_y = 1.0 / ((1.0 + bond_coupon_rate / QUARTERS_PER_YEAR) ** quarters_idx)
        v_pool = float((tot_coll @ df_y).mean())

    imp_cost = float(pv_coll.mean() - v_pool)
    cov_mat = np.cov(np.vstack([pv_A, pv_B, pv_E]), ddof=1)
    var_p = float(cov_mat.sum())
    std_p = float(np.sqrt(var_p))
    cov_p = cov_mat.sum(axis=1)
    lam_var = imp_cost / var_p

    val_A = float(pv_A.mean() - lam_var * cov_p[0])
    val_B = float(pv_B.mean() - lam_var * cov_p[1])
    val_E = float(pv_E.mean() - lam_var * cov_p[2])
    eq_notional = (N_BONDS * FACE_VALUE) - class_a_notional - class_b_notional

    return {
        "A_Notional": class_a_notional,
        "B_Notional": class_b_notional,
        "E_Notional": eq_notional,
        "E[PV_Pool]": float(pv_coll.mean()),
        "V_Pool": v_pool,
        "Implicit_Cost": imp_cost,
        "Pool_Std": std_p,
        "Lambda_Std": imp_cost / std_p,
        "E[PV_A]": float(pv_A.mean()),
        "E[PV_B]": float(pv_B.mean()),
        "E[PV_E]": float(pv_E.mean()),
        "Cov_A": float(cov_p[0]),
        "Cov_B": float(cov_p[1]),
        "Cov_E": float(cov_p[2]),
        "Val_A": val_A,
        "Val_B": val_B,
        "Val_E": val_E,
        "Def_Prob_A": float(def_A.mean()),
        "Def_Prob_B": float(def_B.mean()),
        "Def_Prob_Any": float(def_any.mean()),
    }


# Also display the Par-Calibrated Case (8.0% Bond Coupon, Bonds Valued at $100 MM Par)
par_case = evaluate_cdo_structure(
    class_a_notional=CLASS_A_NOTIONAL,
    class_a_coupon=CLASS_A_COUPON,
    class_b_notional=CLASS_B_NOTIONAL,
    class_b_coupon=CLASS_B_COUPON,
    bond_coupon_rate=0.08,
    bond_valuation_mode="par",
)

print("\n" + "=" * 100)
print("ALTERNATIVE PARAMETER CHECK (PER PS): 8.0% BOND COUPON WITH BONDS PRICED AT $100 MM PAR")
print("=" * 100)
print(
    f"Expected Pool PV at 4% RF: ${par_case['E[PV_Pool]']/1e6:.4f} MM | "
    f"Par Valuation: ${par_case['V_Pool']/1e6:.4f} MM | "
    f"Implicit Cost of Risk: ${par_case['Implicit_Cost']/1e6:.4f} MM | "
    f"Lambda_Std: {par_case['Lambda_Std']:.6f}"
)
par_table_df = pd.DataFrame(
    {
        "Notional ($MM)": [15.0, 30.0, 55.0, 100.0],
        "E[PV] at 4% RF ($MM)": [
            par_case["E[PV_A]"] / 1e6,
            par_case["E[PV_B]"] / 1e6,
            par_case["E[PV_E]"] / 1e6,
            par_case["E[PV_Pool]"] / 1e6,
        ],
        "Risk Charge ($MM)": [
            (par_case["E[PV_A]"] - par_case["Val_A"]) / 1e6,
            (par_case["E[PV_B]"] - par_case["Val_B"]) / 1e6,
            (par_case["E[PV_E]"] - par_case["Val_E"]) / 1e6,
            par_case["Implicit_Cost"] / 1e6,
        ],
        "Lintner Value ($MM)": [
            par_case["Val_A"] / 1e6,
            par_case["Val_B"] / 1e6,
            par_case["Val_E"] / 1e6,
            (par_case["Val_A"] + par_case["Val_B"] + par_case["Val_E"]) / 1e6,
        ],
        "Value / Par (%)": [
            (par_case["Val_A"] / 15e6) * 100,
            (par_case["Val_B"] / 30e6) * 100,
            (par_case["Val_E"] / 55e6) * 100,
            100.0,
        ],
    },
    index=["Class A", "Class B", "Residual (Equity)", "Total CDO"],
)
print(par_table_df.round(4))


# ------------------------------------------------------------------------------
# Step 15 (Part II Step 6 - BONUS): Optimize the Structure Holding Tranche A
# Principal at $15 MM
# ------------------------------------------------------------------------------

opt_rows = []
for b_mm in range(0, 85, 2):
    b_not = b_mm * 1_000_000
    res_base = evaluate_cdo_structure(
        class_a_notional=15_000_000,
        class_a_coupon=CLASS_A_COUPON,
        class_b_notional=b_not,
        class_b_coupon=CLASS_B_COUPON,
        bond_coupon_rate=COUPON_RATE,
        bond_valuation_mode="yield",
    )
    res_par = evaluate_cdo_structure(
        class_a_notional=15_000_000,
        class_a_coupon=CLASS_A_COUPON,
        class_b_notional=b_not,
        class_b_coupon=CLASS_B_COUPON,
        bond_coupon_rate=0.08,
        bond_valuation_mode="par",
    )
    opt_rows.append(
        {
            "Class B Notional ($MM)": b_mm,
            "Residual Notional ($MM)": res_base["E_Notional"] / 1e6,
            "CDO Default Freq (%)": res_base["Def_Prob_Any"] * 100,
            "Val_A ($MM)": res_base["Val_A"] / 1e6,
            "Val_B ($MM)": res_base["Val_B"] / 1e6,
            "Class B Premium Val_B - B ($MM)": (res_base["Val_B"] - b_not) / 1e6,
            "Residual Lintner Val_E ($MM)": res_base["Val_E"] / 1e6,
            "Residual Value / Notional (%)": (res_base["Val_E"] / res_base["E_Notional"]) * 100,
            "Expected Residual Val in Non-Default ($MM)": (
                res_base["Val_E"] * (1.0 - res_base["Def_Prob_Any"]) / 1e6
            ),
            "ParCase Val_E ($MM)": res_par["Val_E"] / 1e6,
            "ParCase Val_E / Notional (%)": (res_par["Val_E"] / res_par["E_Notional"]) * 100,
            "ParCase Class B Premium ($MM)": (res_par["Val_B"] - b_not) / 1e6,
        }
    )

opt_df = pd.DataFrame(opt_rows)
opt_df.to_csv("sensitivity/lintner_optimization_results.csv", index=False)

print("\n" + "=" * 100)
print("PART II - STEP 6 (BONUS): STRUCTURE OPTIMIZATION HOLDING CLASS A PRINCIPAL AT $15 MM")
print("=" * 100)
print(
    opt_df[opt_df["Class B Notional ($MM)"].isin([0, 10, 20, 28, 30, 35, 40, 48, 50, 60, 66, 70, 80])]
    .round(4)
    .to_string(index=False)
)

# Identify key optimal structures across standard structured-finance criteria
zero_def_df = opt_df[opt_df["CDO Default Freq (%)"] == 0.0]
max_zero_def_row = zero_def_df.iloc[-1]
max_b_premium_row = opt_df.loc[opt_df["Class B Premium Val_B - B ($MM)"].idxmax()]
par_feasible_df = opt_df[opt_df["Class B Premium Val_B - B ($MM)"] >= 0.0]
max_par_leverage_row = par_feasible_df.iloc[-1]
max_raw_residual_row = opt_df.loc[opt_df["Residual Lintner Val_E ($MM)"].idxmax()]

print("\nSummary of Optimal Tranche B Sizing (Holding Class A = $15.00 MM):")
print("------------------------------------------------------------------")
print(
    f"1. Max Debt Structure with Zero Defaults (Default Indicator = 0%): "
    f"Class B = ${max_zero_def_row['Class B Notional ($MM)']:.0f} MM, "
    f"Residual Notional = ${max_zero_def_row['Residual Notional ($MM)']:.0f} MM, "
    f"Residual Value = ${max_zero_def_row['Residual Lintner Val_E ($MM)']:.4f} MM "
    f"(Value/Notional = {max_zero_def_row['Residual Value / Notional (%)']:.2f}%)"
)
print(
    f"2. Max Tranche B Value Creation (Max [Val_B - B_Notional] before default erosion): "
    f"Class B = ${max_b_premium_row['Class B Notional ($MM)']:.0f} MM, "
    f"Class B Premium = +${max_b_premium_row['Class B Premium Val_B - B ($MM)']:.4f} MM, "
    f"Default Rate = {max_b_premium_row['CDO Default Freq (%)']:.2f}%, "
    f"Residual Value = ${max_b_premium_row['Residual Lintner Val_E ($MM)']:.4f} MM"
)
print(
    f"3. Max Par-Feasible Leverage (Largest Class B where Val_B >= Par): "
    f"Class B = ${max_par_leverage_row['Class B Notional ($MM)']:.0f} MM, "
    f"Residual Notional = ${max_par_leverage_row['Residual Notional ($MM)']:.0f} MM, "
    f"Default Rate = {max_par_leverage_row['CDO Default Freq (%)']:.2f}%, "
    f"Residual Value = ${max_par_leverage_row['Residual Lintner Val_E ($MM)']:.4f} MM"
)
print(
    f"4. Max Unconstrained Dollar Residual Value (Val_E): "
    f"Class B = ${max_raw_residual_row['Class B Notional ($MM)']:.0f} MM, "
    f"Residual Notional = ${max_raw_residual_row['Residual Notional ($MM)']:.0f} MM, "
    f"Residual Value = ${max_raw_residual_row['Residual Lintner Val_E ($MM)']:.4f} MM "
    f"(Value/Notional = {max_raw_residual_row['Residual Value / Notional (%)']:.2f}%)"
)

# Plot Optimization Curves
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(
    opt_df["Class B Notional ($MM)"],
    opt_df["Residual Lintner Val_E ($MM)"],
    label="Residual Lintner Value ($MM)",
    linewidth=2,
)
ax1.plot(
    opt_df["Class B Notional ($MM)"],
    opt_df["Class B Premium Val_B - B ($MM)"] * 10,
    label="10x Class B Market Premium [Val_B - B] ($MM)",
    linestyle="--",
    linewidth=2,
)
ax1.set_xlabel("Class B Principal ($MM) [Holding Class A = $15 MM]")
ax1.set_ylabel("Value ($MM)")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(
    opt_df["Class B Notional ($MM)"],
    opt_df["CDO Default Freq (%)"],
    color="crimson",
    linestyle=":",
    linewidth=2,
    label="CDO Default Indicator Freq (%)",
)
ax2.set_ylabel("CDO Default Probability (%)", color="crimson")

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
plt.title("CDO Structure Optimization (Holding Class A = $15 MM)")
plt.tight_layout()
plt.savefig("distributions/lintner_optimization.png", dpi=300, bbox_inches="tight")
plt.close()

if sys.stdin.isatty():
    while True:

        user_input = input(
            "\nEnter a case number (1-1000), or 'q' to quit: "
        )

        if user_input.lower() == "q":
            break

        try:
            case_num = int(user_input)
            view_case(case_num)

        except ValueError:
            print("Please enter a valid case number from 1 to 1000.")