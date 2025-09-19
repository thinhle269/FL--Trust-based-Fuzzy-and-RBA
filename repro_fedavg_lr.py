# -*- coding: utf-8 -*-
import argparse, json, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             precision_recall_fscore_support, accuracy_score)

def split_by_user(df, seed=42, train_ratio=0.8):
    rng = np.random.default_rng(seed)
    users = np.array(sorted(df['User ID'].unique()))
    rng.shuffle(users)
    cut = int(train_ratio * len(users))
    return set(users[:cut]), set(users[cut:])

def hash_bucket(u, n=10):
    return hash(str(u)) % n

def policy_tradeoffs(trust, y_true, out_base):
    taus = np.linspace(0.05, 0.95, 19)
    ch, prec, rec, f1 = [], [], [], []
    for t in taus:
        yhat = (trust < t).astype(int)
        ch.append(float(np.mean(yhat)))
        p,r,f,_ = precision_recall_fscore_support(y_true, yhat, average='binary', zero_division=0)
        prec.append(p); rec.append(r); f1.append(f)
    plt.figure(figsize=(8,5))
    plt.plot(taus, ch, label="Challenge rate")
    plt.plot(taus, prec, label="Precision")
    plt.plot(taus, rec, label="Recall")
    plt.plot(taus, f1, label="F1")
    plt.xlabel("Threshold τ"); plt.ylabel("Rate / Score"); plt.title("Policy trade-offs")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{out_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{out_base}.pdf", bbox_inches='tight')
    plt.close()

def plot_roc(prob_list, labels, y_true, out_base):
    plt.figure(figsize=(7,6))
    for prob, lab in zip(prob_list, labels):
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        plt.plot(fpr, tpr, label=f"{lab} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],'--', label="Chance")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC comparison")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(f"{out_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{out_base}.pdf", bbox_inches='tight')
    plt.close()

def plot_cm(cm, title, out_base):
    plt.figure(figsize=(4.8,4.2))
    ax = plt.gca(); im = ax.imshow(cm)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Pred 0","Pred 1"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["True 0","True 1"])
    ax.set_title(title)
    vmax = cm.max()
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i,j] > 0.6*vmax else "black"
            ax.text(j, i, f"{int(cm[i,j])}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f"{out_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{out_base}.pdf", bbox_inches='tight')
    plt.close()

def fl_train_lr(clients, rounds=50, local_epochs=1, lr=1e-2, batch=32, pos_weight=1.0):
    d = clients[0][0].shape[1]
    w_global = np.zeros(d, dtype=float); b_global = 0.0
    def one_round(w, b):
        new_ws, new_bs, sizes = [], [], []
        for (X, y) in clients:
            w_loc = w.copy(); b_loc = float(b)
            n = len(X); idx = np.arange(n)
            rng = np.random.default_rng()
            for _ in range(local_epochs):
                rng.shuffle(idx)
                for s in range(0, n, batch):
                    sl = idx[s:s+batch]; xb = X[sl]; yb = y[sl]
                    z = xb.dot(w_loc) + b_loc; p = 1/(1+np.exp(-z))
                    weights = np.where(yb==1.0, pos_weight, 1.0)
                    diff = (p - yb) * weights
                    grad_w = xb.T.dot(diff) / len(xb)
                    grad_b = float(np.mean(diff))
                    w_loc -= lr * grad_w; b_loc -= lr * grad_b
            new_ws.append(w_loc); new_bs.append(b_loc); sizes.append(n)
        total = float(np.sum(sizes))
        w_new = np.zeros_like(w); b_new = 0.0
        for wi, bi, ni in zip(new_ws, new_bs, sizes):
            w_new += wi * (ni/total); b_new += bi * (ni/total)
        return w_new, b_new
    hist = []
    for r in range(1, rounds+1):
        w_global, b_global = one_round(w_global, b_global)
        hist.append((r, w_global.copy(), b_global))
    return (w_global, b_global), hist

def evaluate_lr(w, b, X, y_true):
    z = X.dot(w) + b
    prob = 1/(1+np.exp(-z)); pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, pred)
    auc = roc_auc_score(y_true, prob); acc = accuracy_score(y_true, pred)
    p,r,f,_ = precision_recall_fscore_support(y_true, pred, average='binary', zero_division=0)
    return dict(prob=prob, pred=pred, cm=cm, auc=float(auc), acc=float(acc), p=float(p), r=float(r), f=float(f))

def main(csv, out_dir, clients=10, rounds=50, local_epochs=1, seed=42):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv)
    # harmonize names if needed
    if 'f_ipClean' in df.columns and 'f_ip_clean' not in df.columns: df['f_ip_clean']=df['f_ipClean']
    feat_all = ['f_device','f_country','f_asn','f_ip_clean','f_success']
    ycol='silver_label'
    train_u, test_u = split_by_user(df, seed=seed, train_ratio=0.8)
    tr = df[df['User ID'].isin(train_u)].copy(); te = df[df['User ID'].isin(test_u)].copy()
    # policy figure
    policy_tradeoffs(te['fuzzy_trust'].values, te[ycol].astype(int).values, out/'fig_policy_tradeoffs')
    # centralized LR
    scaler = StandardScaler().fit(tr[feat_all].values)
    Xtr = scaler.transform(tr[feat_all].values); ytr = tr[ycol].astype(int).values
    Xte = scaler.transform(te[feat_all].values); yte = te[ycol].astype(int).values
    cen = LogisticRegression(max_iter=400, solver='lbfgs').fit(Xtr, ytr)
    cen_prob = cen.predict_proba(Xte)[:,1]
    # heuristic baseline
    heur_score = 1.0 - te['f_ip_clean'].values; heur_pred = (heur_score >= 0.5).astype(int)
    # FL-LR (proxy): hash bucket by user
    def bucket(u): return hash(str(u)) % clients
    tr_b = tr.copy(); tr_b['bucket'] = tr_b['User ID'].apply(bucket)
    clis = []
    for b in range(clients):
        part = tr_b[tr_b['bucket']==b]
        if len(part)==0: continue
        Xc = scaler.transform(part[feat_all].values); yc = part[ycol].astype(float).values
        clis.append((Xc, yc))
    pos = float(np.sum(ytr==1)); neg = float(np.sum(ytr==0)); pos_weight = neg/max(pos,1.0)
    (w,b), hist = fl_train_lr(clis, rounds=rounds, local_epochs=local_epochs, lr=1e-2, batch=32, pos_weight=pos_weight)
    # eval
    res_fl = evaluate_lr(w,b, Xte, yte)
    # figures
    plot_roc([heur_score, cen_prob, res_fl['prob']],
             ["Heuristic IP rule","Centralized LR","FedAvg-LR (proxy)"], yte, out/'fig_roc_comparison')
    from sklearn.metrics import confusion_matrix
    import numpy as np
    cm_heur = confusion_matrix(yte, heur_pred); cm_cen = confusion_matrix(yte, (cen_prob>=0.5).astype(int))
    def plot_cm(cm, title, out_base):
        plt.figure(figsize=(4.8,4.2)); ax=plt.gca(); im=ax.imshow(cm)
        ax.set_xticks([0,1]); ax.set_xticklabels(["Pred 0","Pred 1"])
        ax.set_yticks([0,1]); ax.set_yticklabels(["True 0","True 1"])
        ax.set_title(title); vmax=cm.max()
        for i in range(2):
            for j in range(2):
                color="white" if cm[i,j]>0.6*vmax else "black"
                ax.text(j,i, f"{int(cm[i,j])}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")
        plt.colorbar(im, fraction=0.046, pad=0.04); plt.tight_layout()
        plt.savefig(f"{out_base}.png", dpi=300, bbox_inches='tight'); plt.savefig(f"{out_base}.pdf", bbox_inches='tight'); plt.close()
    plot_cm(cm_heur, "Heuristic IP rule", out/'cm_heur')
    plot_cm(cm_cen,  "Centralized LR",   out/'cm_central')
    plot_cm(res_fl['cm'], "FedAvg-LR (proxy)", out/'cm_fedavg_lr')
    # training curves (validation proxy): use every 5 rounds
    # For brevity, skip validation curves here
    # tables
    def row(name, prob, y_true):
        pred=(prob>=0.5).astype(int); cm=confusion_matrix(y_true, pred)
        p,r,f,_=precision_recall_fscore_support(y_true, pred, average='binary', zero_division=0)
        return dict(Model=name, AUC=roc_auc_score(y_true, prob), Accuracy=accuracy_score(y_true, pred),
                    Precision=p, Recall=r, F1=f, TN=cm[0,0], FP=cm[0,1], FN=cm[1,0], TP=cm[1,1])
    t2 = pd.DataFrame([row("Heuristic IP rule", heur_score, yte),
                       row("Centralized LR", cen_prob, yte),
                       row("FedAvg-LR (proxy)", res_fl['prob'], yte)]).round(4)
    t2.to_csv(out/'Table02_key_metrics.csv', index=False)
    # ablations
    def central_metrics(cols):
        Xtr = scaler.fit_transform(tr[cols].values); Xte = scaler.transform(te[cols].values)
        m = LogisticRegression(max_iter=400, solver='lbfgs').fit(Xtr, ytr)
        prob = m.predict_proba(Xte)[:,1]; pred=(prob>=0.5).astype(int)
        cm=confusion_matrix(yte, pred); p,r,f,_=precision_recall_fscore_support(yte, pred, average='binary', zero_division=0)
        return dict(AUC=roc_auc_score(yte, prob), Accuracy=accuracy_score(yte, pred), Precision=p, Recall=r, F1=f)
    full = ['f_device','f_country','f_asn','f_ip_clean','f_success']
    rm_ip = ['f_device','f_country','f_asn','f_success']
    rm_su = ['f_device','f_country','f_asn','f_ip_clean']
    rm_both=['f_device','f_country','f_asn']
    only_direct=['f_ip_clean','f_success']
    rows=[("Full (all 5)", full), ("Removal: - f_ip_clean", rm_ip), ("Removal: - f_success", rm_su),
          ("Removal: - f_ip_clean & - f_success", rm_both), ("Only direct-rule features", only_direct)]
    out_rows=[]
    for name, cols in rows:
        met=central_metrics(cols)
        out_rows.append(dict(Setting=name, Features=", ".join(cols), **{k:round(float(met[k]),4) for k in met}))
    pd.DataFrame(out_rows).to_csv(out/'Table03_ablation_key_metrics.csv', index=False)
    print("Saved figures and tables to", out.resolve())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="outputs_lr")
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--local-epochs", type=int, default=1)
    args = ap.parse_args()
    main(args.csv, args.out, args.clients, args.rounds, args.local_epochs)
