# Late-Payment Risk Model

A machine learning project that ranks accounts-receivable (AR) invoices by their risk of being paid late, so a finance team knows which customers to chase first.

![Feature importance](feature_importance.png)

## Business question

I work in finance (AP, AR, GL, reconciliations). Chasing late payments is a daily problem: with limited time, the AR team either chases everyone or chases no one.

**Question:** Can a model tell us, at the time an invoice is issued, which invoices are most likely to be paid late — so we can prioritise collection calls?

## Dataset

- `invoices.csv` — **600 synthetic invoices** from 10 fictional customers, dated Jan 2024 to Nov 2025.
- The data is **not real company data**. It was generated for practice, with built-in patterns (some customers pay late more often; larger invoices and December invoices are slightly riskier) plus random noise, so it behaves like real AR data without exposing anything confidential.
- Columns: `invoice_id`, `customer`, `invoice_date`, `amount`, `payment_terms_days`, `due_date`, `payment_date`, `paid_late` (Yes/No).
- 32.7% of invoices are paid late.

## Method

1. **Explore** the data with pandas: late rate overall and per customer.
2. **Feature engineering:** invoice month, a December flag, one-hot encoding of the customer. `payment_date` is deliberately excluded — it is not known when we predict.
3. **Split** 80% training / 20% held-out test (`random_state=42`).
4. **Model:** Random Forest classifier (scikit-learn), 200 trees, `max_depth=4` to limit overfitting.
5. **Evaluate** on the held-out test set.
6. **Use** predicted probabilities to rank invoices by risk, and inspect feature importance.

## Results (held-out test set, 120 invoices)

| Metric | Value | Note |
|---|---|---|
| Accuracy | 0.68 | Baseline (always predict "not late") = 0.67 |
| ROC AUC | 0.73 | How well the model ranks late above on-time invoices (0.5 = random, 1.0 = perfect) |
| Top-15 riskiest invoices | 10 of 15 actually late | vs. ~5 of 15 expected if picked at random |

**Interpretation:** accuracy is modest because the practice data is noisy — even the riskiest customer pays on time 30% of the time. The useful output is the **ranking**: calling the top-ranked invoices first catches twice as many real late payers as a random list.

**Main drivers** (feature importance): invoice amount, whether the customer is Delta Foods, invoice month, and whether the customer is Iris Design. Importance values sum to 1.0 and show which inputs the model relied on — they are not accuracy or probability figures.

## Why this matters to a company

- **Focus collection effort.** Instead of chasing every overdue account, the AR team starts each week with a risk-ranked list. In this test, calling the top 15 caught 10 real late payers — roughly double a random pick.
- **Improve cash flow.** Earlier contact with high-risk customers means faster payment and less money tied up in receivables (lower DSO).
- **Catch problems early.** A customer whose risk score rises month after month is a warning sign before the account becomes a bad debt.
- **Explainable, not a black box.** Feature importance shows *why* an invoice is flagged (customer, amount, timing), so finance staff can check the reasoning and management can trust it.
- **Low cost to start.** Built with free, open-source tools on ordinary invoice data that every finance system already has. No new software to buy.

## Limitations

- Synthetic data with only 5 input features. Real AR data would add credit terms, payment history, industry, disputes, and so on, and would likely score higher.
- Small dataset (600 rows). Results vary with a different train/test split.
- The model is trained on the customers in the data; a brand-new customer would fall back to amount, terms and month only.
- No cost-weighting: a missed late payer and a false alarm are treated as equally bad, which is not true in practice.

## How to run

1. Install Python 3 with `pandas`, `scikit-learn`, `matplotlib` and `jupyter` (Anaconda includes all of these).
2. Put `invoices.csv` and `late_payment_model.ipynb` in the same folder.
3. Open the notebook in Jupyter and run the cells from top to bottom (Shift + Enter).
4. The last cells save the trained model with `joblib` and show how to score a new invoice.

## Files

- `late_payment_model.ipynb` — the full notebook, with notes in English and Chinese
- `invoices.csv` — synthetic practice data
- `feature_importance.png` — chart of the model's feature importance

## Tools

Python · pandas · scikit-learn · matplotlib · Jupyter
  

## Tools

Python · pandas · scikit-learn · matplotlib · Jupyter
