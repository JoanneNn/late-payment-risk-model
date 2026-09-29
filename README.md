# late-payment-risk-model
Machine learning model to rank AR invoices by risk of late payment (Python, scikit-learn)

# Late-Payment Risk Model

A machine learning project that ranks accounts-receivable invoices by risk of late payment,
so a finance team knows which customers to chase first.

## Background
I work in finance (AP, AR, GL, reconciliations). Chasing late payments is a daily problem.
This project asks: can a model tell us in advance which invoices are likely to be late?

## What I did
- Explored 600 invoices (practice data, 10 customers) with pandas
- Feature engineering: invoice month, December flag, one-hot encoded customers
- Trained a Random Forest classifier (scikit-learn), tuned max_depth to reduce overfitting
- Used predicted probabilities to build a risk-ranked invoice list
- Checked feature importance to see what drives late payment

## Results
- Accuracy ~0.68 vs. a baseline of 0.67 — the practice data is noisy, so accuracy is modest
- The useful output is the ranking: the top-15 riskiest invoices are mostly real late payers
- Main drivers: invoice amount, customer identity, invoice month

## Files
- `late_payment_model.ipynb` — the full notebook, with notes in English and Chinese
- `invoices.csv` — practice data (synthetic, not real company data)

## Tools
Python · pandas · scikit-learn · Jupyter
