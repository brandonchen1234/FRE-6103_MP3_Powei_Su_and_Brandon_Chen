import numpy as np
import pandas as pd
from scipy.stats import norm


class CDOSimulator:
    def __init__(self, seed=42):
        # 1. Initialize Parameters
        self.n_sim = 1000
        self.n_bonds = 10
        self.t_years = 5
        self.n_quarters = self.t_years * 4

        # Bond Parameters
        self.face_value = 10_000_000
        self.coupon_rate = 0.06
        self.p_default_annual = 0.04
        self.lgd = 0.60
        self.recovery_rate = 1 - self.lgd
        self.rho = 0.20

        # CDO Tranche Parameters
        self.notional_A = 20_000_000
        self.coupon_A = 0.02
        self.notional_B = 10_000_000
        self.coupon_B = 0.04

        # Set fixed random seed for reproducibility (fixed random numbers)
        np.random.seed(seed)

        # Annual default intensity (hazard rate)
        self.lambda_ = -np.log(1 - self.p_default_annual)

    def run_simulation(self):
        """Simulates correlated defaults and generates cash flows for all 1000 cases."""

        # --- TASK 1: Simulate Correlated Defaults (Gaussian Copula / BIS Model) ---
        # 1-Factor Gaussian Copula: Z_i = sqrt(rho)*M + sqrt(1-rho)*epsilon_i
        M = np.random.normal(size=(self.n_sim, 1))  # Systemic factor
        epsilon = np.random.normal(size=(self.n_sim, self.n_bonds))  # Idiosyncratic factor

        Z = np.sqrt(self.rho) * M + np.sqrt(1 - self.rho) * epsilon

        # Map normal variables to uniform variables
        U = norm.cdf(Z)

        # Map uniform variables to default times using Exponential distribution
        tau = -np.log(1 - U) / self.lambda_

        # Determine the exact quarter of default
        q_default = np.ceil(tau * 4).astype(int)

        # --- GENERATE COLLATERAL CASH FLOWS ---
        # Initialize 3D array: (1000 simulations, 10 bonds, 20 quarters)
        quarters = np.arange(1, self.n_quarters + 1)

        # Masks for cash flow logic
        is_alive = quarters[None, None, :] < q_default[:, :, None]
        is_default_q = quarters[None, None, :] == q_default[:, :, None]
        is_maturity = quarters == self.n_quarters

        # Calculate CF components
        coupon_cf = is_alive * (self.face_value * self.coupon_rate / 4)
        recovery_cf = is_default_q * (self.face_value * self.recovery_rate)

        # Principal is paid at maturity (Quarter 20) only if the bond is still alive
        principal_cf = is_maturity[None, None, :] * is_alive * self.face_value

        # Total bond cash flows: shape (1000, 10, 20)
        self.bond_cfs = coupon_cf + recovery_cf + principal_cf

        # Aggregate collateral cash flows per quarter: shape (1000, 20)
        self.total_collateral_cf = self.bond_cfs.sum(axis=1)

        # --- TASK 2: Apply Waterfall Rules ---
        # Calculate promised cash flows per tranche (Assuming Bullet Maturity)
        promised_A = np.full(self.n_quarters, self.notional_A * self.coupon_A / 4)
        promised_A[-1] += self.notional_A  # Add principal at maturity

        promised_B = np.full(self.n_quarters, self.notional_B * self.coupon_B / 4)
        promised_B[-1] += self.notional_B  # Add principal at maturity

        # Waterfall Distribution (No carry-forwards, distributed as earned)
        self.cf_A = np.minimum(self.total_collateral_cf, promised_A)

        rem_after_A = self.total_collateral_cf - self.cf_A
        self.cf_B = np.minimum(rem_after_A, promised_B)

        self.cf_Equity = rem_after_A - self.cf_B

    def generate_statistics_report(self):
        """TASK 3: Generates a statistical analysis of the total cash flows over the 5 years."""
        # Sum cash flows across all 20 quarters for each simulation
        total_cf_collateral = self.total_collateral_cf.sum(axis=1)
        total_cf_A = self.cf_A.sum(axis=1)
        total_cf_B = self.cf_B.sum(axis=1)
        total_cf_Equity = self.cf_Equity.sum(axis=1)

        data = {
            "Total Collateral CF": total_cf_collateral,
            "Class A (Senior) CF": total_cf_A,
            "Class B (Mezz) CF": total_cf_B,
            "Equity (Residual) CF": total_cf_Equity
        }

        df = pd.DataFrame(data)

        # Calculate key statistics
        stats = df.describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
        stats = stats[['mean', 'std', 'min', '1%', '5%', '25%', '50%', '75%', '95%', '99%', 'max']]

        print("\n" + "=" * 85)
        print(" TASK 3: STATISTICAL ANALYSIS OF TOTAL CASH FLOWS (OVER 5 YEARS)")
        print(" (Values in $ Millions)")
        print("=" * 85)
        # Format the output into millions for readability
        print((stats / 1_000_000).round(2).to_string())
        print("=" * 85)

    def view_case(self, case_num):
        """Displays quarter-by-quarter cash flows for a specific selected case (1 to 1000)."""
        if not (1 <= case_num <= self.n_sim):
            raise ValueError(f"Case number must be between 1 and {self.n_sim}")

        idx = case_num - 1  # 0-based indexing

        print(f"\n" + "=" * 85)
        print(f" QUARTERLY CASH FLOWS FOR CASE #{case_num} (in $)")
        print("=" * 85)

        # 1. Display individual bond cash flows
        bond_df = pd.DataFrame(self.bond_cfs[idx].T,
                               index=[f"Q{i}" for i in range(1, 21)],
                               columns=[f"Bond {i}" for i in range(1, 11)])
        print("\n--- 10 Collateral Bonds Cash Flows ---")
        print(bond_df.round(0).applymap(lambda x: f"{x:,.0f}"))

        # 2. Display CDO Waterfall cash flows
        cdo_df = pd.DataFrame({
            "Total Collateral": self.total_collateral_cf[idx],
            "Class A Paid": self.cf_A[idx],
            "Class B Paid": self.cf_B[idx],
            "Equity Paid": self.cf_Equity[idx]
        }, index=[f"Q{i}" for i in range(1, 21)])

        print("\n--- CDO Tranche Waterfall ---")
        print(cdo_df.round(0).applymap(lambda x: f"{x:,.0f}"))


if __name__ == "__main__":
    # Initialize and run the model
    cdo_model = CDOSimulator(seed=123)
    cdo_model.run_simulation()

    # Print the statistical report (Task 3)
    cdo_model.generate_statistics_report()

    # Print a specific scenario so the user can see exact mechanics (Tasks 1 & 2)
    # You can change the case number here (1 to 1000)
    cdo_model.view_case(case_num=1)