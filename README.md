# Mini-Project 3: Simplified CDO Analysis

**FRE 6103 – Valuation for Financial Engineers** | Prof. David Shimko | Sep 2026

## **The underlying assets**

A bank has placed 10 speculative grade corporate bonds in a CDO it is looking to sell. Each bond is 5 years in duration, makes quarterly coupon payments at an annual coupon rate of 6%, and the annual default probability is 4%. The LGD is 60% for all bonds. The market YTM on the bonds is 9%, and the risk-free interest rate is 1% per year. The “correlation” between any two bonds’ default dates is equal to a constant of 0.20. The bonds’ face values are $10 MM each.

## **The CDO**

The structuring group has proposed two PACs for the CDO. The Class A tranche has an initial notional value of $20 MM and a coupon rate of 2% per year. The Class B tranche has a notional value of $10 MM and a coupon rate of 4% per year. The bank will retain the residuals as equity.

## **The Waterfall**

Collateral interest cash flows are used first to pay Class A interest, then Class B interest, with any remaining interest distributed to the bank as the residual equity holder. Recovery and principal cash flows are applied sequentially to Class A principal, then Class B principal, with any remaining amount distributed to the bank. There are no carry-forwards, and all cash flows are distributed as earned. Tranche coupons are calculated on beginning-of-quarter outstanding principal balances. Because the project description does not specify a detailed principal amortization schedule, this principal allocation is treated as a modeling assumption.

> *Suggestion: Compute promised cash flows of the collateral, adjust the cash flows for default times, and distribute the cash flows according to the waterfall.*

## **Tasks**

* **Task 1.** Build a simulation model of the aggregate quarterly cash flows of the portfolio, as a function of the quarterly cash flows. Be sure to correlate the simulated default dates.
* **Task 2.** Apply the waterfall rules to determine the cash flow patterns to each class.
* **Task 3.** Provide a statistical analysis of the cash flows (both the collateral and the classes) in a report suitable for a bank client. Be sure to include probability distributions of the outcomes, and key sensitivities to the parameters.
