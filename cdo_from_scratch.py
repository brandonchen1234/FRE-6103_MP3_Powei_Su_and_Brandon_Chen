import os
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

# Class A (safest and lowest yield)
CLASS_A_NOTIONAL = 20_000_000
CLASS_A_COUPON = 0.02

# Class B (riskier but higher yield)
CLASS_B_NOTIONAL = 10_000_000
CLASS_B_COUPON = 0.04

# Class C (Equity Tranche gets whatever is left over)

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
# Step 3: Generate Random Normals
# -----------------------------

np.random.seed(SEED)

# Generate 1,000 sets of 10 independent random numbers from a standard normal distribution
independent_normals = np.random.normal(
    size=(N_SIM, N_BONDS)
)

# print("\nIndependent normals shape:")
# print(independent_normals.shape)

# Multiply the independent normals by the Cholesky matrix to get correlated matrix
correlated_normals = independent_normals @ cholesky_matrix.T

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
print("TASK 3: TOTAL 5-YEAR CASH FLOW STATISTICS")
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
plt.show()

plt.figure()

plt.hist(
    total_cf_A / 1_000_000,
    bins=30
)

plt.xlabel("Total 5-Year Cash Flow ($MM)")
plt.ylabel("Frequency")
plt.title("Distribution of Class A Cash Flows")
plt.savefig("distributions/class_A_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure()

plt.hist(
    total_cf_B / 1_000_000,
    bins=30
)

plt.xlabel("Total 5-Year Cash Flow ($MM)")
plt.ylabel("Frequency")
plt.title("Distribution of Class B Cash Flows")
plt.savefig("distributions/class_B_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure()

plt.hist(
    total_cf_equity / 1_000_000,
    bins=30
)

plt.xlabel("Total 5-Year Cash Flow ($MM)")
plt.ylabel("Frequency")
plt.title("Distribution of Equity Cash Flows")
plt.savefig("distributions/equity_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

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

    # Independent normals -> correlated normals
    independent_normals = np.random.normal(
        size=(N_SIM, N_BONDS)
    )

    correlated_normals = independent_normals @ L.T

    # Normal -> Uniform
    uniforms = norm.cdf(correlated_normals)

    # Uniform -> default time
    default_times = (
        np.log(1 - uniforms)
        / np.log(1 - pd_annual)
    )

    default_quarters = np.ceil(
        default_times * QUARTERS_PER_YEAR
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

            dq = default_quarters[sim, bond]

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

plt.show()

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

plt.show()

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