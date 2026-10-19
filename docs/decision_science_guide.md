# Decision-science extension: what each module means

These five ideas answer different questions. Keeping them separate makes the
project easier to explain and prevents claims that the evidence cannot support.

## 1. Price prediction

The existing LightGBM and PyTorch models predict the selling price of one
equipment row from its date, age, usage, region, category, and attributes.

Example: "What may this 8-year-old excavator sell for?"

## 2. Market forecasting

`src/market_forecasting.py` compresses many auctions into one median price per
month. It then uses earlier months to predict later months.

Example: "What may the typical market price be next month?"

The latest 12 historical months are hidden during training. This backtest is the
forecasting equivalent of a practice exam: the model predicts months it did not
see while fitting.

## 3. Hypothesis testing

The R analysis tests two null hypotheses:

- ANOVA: category mean log prices are equal;
- Welch test: High- and Low-utilization mean log prices are equal.

A small p-value says the observed difference would be surprising under the null
hypothesis. It does not say the difference is large, useful, or causal. That is
why the report also gives eta-squared and Cohen's d effect sizes.

## 4. Causal sensitivity analysis

High-utilization equipment may differ from Low-utilization equipment in age,
sale year, category, and region. Propensity weighting tries to make the observed
groups more comparable on those measured variables before comparing prices.

This still cannot balance unmeasured mechanical condition, damage, seller
behaviour, or local demand. The result is therefore a sensitivity analysis, not
proof that changing utilization would cause the reported price difference.

## 5. EOQ and inventory planning

EOQ balances two scenario costs:

```text
EOQ = sqrt(2 * annual demand * ordering cost / annual holding cost per unit)
```

Ordering very small quantities creates many ordering cycles. Ordering a very
large quantity increases average inventory and holding cost. EOQ finds the
minimum of this simplified cost curve.

The auction dataset does not contain a dealer's real demand, ordering cost,
holding rate, or supplier lead time. The module therefore uses transaction count
only as a market-activity proxy and reads business assumptions from
`config/inventory_assumptions.example.json`. Replace those values with verified
business inputs before using the result outside a learning demonstration.
