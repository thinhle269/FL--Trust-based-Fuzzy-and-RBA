# FL--Trust-based-Fuzzy-and-RBA
# RBA Fuzzy + Federated Learning (Trust → Silver Labels → FL)

## Huan và Duy 

## Files
- `build_labels.py` — Build fuzzy trust + silver labels from Excel to CSV. (Uploaded original)
- `fl_rba.py` — Federated learner with ANN-BN backbone (Client/Server, FedAvg). (Uploaded original)
- `run_experiments.py` — Run centralized LR, federated ANN-BN, ablations, and save figures. (Uploaded original)
- `plotting.py` — Plot policy trade-offs, ROC, confusion matrices, convergence. (Uploaded original)
- `repro_fedavg_lr.py` — **New**: self-contained Federated LR (proxy) runner to reproduce figures/tables using the CSV.
- `requirements.txt` — Minimal requirements to run both paths.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) If you start from Excel, build fuzzy trust + silver labels
python build_labels.py --xlsx data/rba_test.xlsx --out data/rba_fuzzy_labels.csv

# 2A) Run the original experiments (FL ANN-BN)
python run_experiments.py --csv data/rba_fuzzy_labels.csv --out outputs_ann --device cpu

# 2B) Reproduce the **LR-proxy** results used in the paper (E=1, 10 clients, 50 rounds)
python repro_fedavg_lr.py --csv data/rba_fuzzy_labels.csv --out outputs_lr --clients 10 --rounds 50 --local-epochs 1
```

The LR-proxy script mirrors the analysis in the paper: user-level split, FedAvg with **E=1**, class weighting for imbalance, a shared StandardScaler fit on train users, and the same figures/tables:
- `fig_policy_tradeoffs.(png|pdf)`
- `fig_roc_comparison.(png|pdf)`
- `cm_heur.(png|pdf)`, `cm_central.(png|pdf)`, `cm_fedavg_lr.(png|pdf)`
- `fig_fl_convergence.(png|pdf)`
- `Table02_key_metrics.csv`, `Table03_ablation_key_metrics.csv`
