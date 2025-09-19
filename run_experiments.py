 
import argparse, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support
from fl_rba import Server, Client
import plotting as P

def split_by_user(df, seed=42, train_ratio=0.8):
    rng=np.random.default_rng(seed)
    users=np.array(sorted(df['User ID'].unique())); rng.shuffle(users)
    cut=int(train_ratio*len(users))
    return set(users[:cut]), set(users[cut:])

def hash_bucket(u, n=10): return hash(str(u)) % n

def make_clients(train_df, feat_cols, n_clients=10, batch=32, device='cpu'):
    clients=[]; sizes=[]
    train_df=train_df.copy(); train_df['bucket']=train_df['User ID'].apply(lambda u:hash_bucket(u,n_clients))
    for b in range(n_clients):
        part=train_df[train_df['bucket']==b]
        if len(part)==0: continue
        X=part[feat_cols].values.astype('float32')
        y=part['silver_label'].astype('float32').values
        clients.append(Client(X,y,batch=batch, device=device))
        sizes.append(len(part))
    return clients, sizes

def centralized_lr(train_df, test_df, feat_cols):
    Xtr=train_df[feat_cols].values; ytr=train_df['silver_label'].astype(int).values
    Xte=test_df[feat_cols].values; yte=test_df['silver_label'].astype(int).values
    pipe=make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, solver='lbfgs'))
    pipe.fit(Xtr,ytr)
    prob=pipe.predict_proba(Xte)[:,1]; pred=(prob>=0.5).astype(int)
    cm=confusion_matrix(yte, pred); auc=roc_auc_score(yte, prob)
    p,r,f1,_=precision_recall_fscore_support(yte, pred, average='binary', zero_division=0)
    return dict(prob=prob, pred=pred, cm=cm, auc=auc, p=p, r=r, f1=f1)

def main(csv='data/rba_fuzzy_labels.csv', out='outputs', device='cpu'):
    out=Path(out); out.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(csv)
    feat_all=['f_device','f_country','f_asn','f_ip_clean','f_success']
    train_users, test_users = split_by_user(df, seed=42, train_ratio=0.8)
    tr=df[df['User ID'].isin(train_users)].copy()
    te=df[df['User ID'].isin(test_users)].copy()

    # Policy trade-offs from trust score
    P.plot_policy_tradeoffs(te['fuzzy_trust'].values, te['silver_label'].astype(int).values,
                            out / 'fig_policy_tradeoffs')

    # Centralized LR (upper bound) and heuristic baseline
    cen=centralized_lr(tr, te, feat_all)
    heur_score=1 - te['f_ip_clean'].values
    heur_pred=(heur_score>=0.5).astype(int)

    # FedAvg-LR proxy (optional quick check)
    # (left out for brevity; we focus on the ANN-BN FL next)

    # Federated ANN-BN
    server=Server(in_dim=len(feat_all), device=device)
    clients, sizes = make_clients(tr, feat_all, n_clients=10, batch=32, device=device)
    # Validation set for monitoring = the held-out users (te)
    history=server.run_fedavg(clients, rounds=50, local_epochs=1, lr=1e-2,
                              class_weight=None, val_data=(te[feat_all].values, te['silver_label'].astype(float).values),
                              log_every=5)
    P.plot_training_curves(history, out/'fig_fl_convergence')

    # Final evaluation (FL)
    res_fl=server.evaluate(te[feat_all].values, te['silver_label'].astype(float).values)
    # Confusion matrices (print-safe)
    P.plot_cm(confusion_matrix(te['silver_label'].astype(int).values, heur_pred),
              "Heuristic IP rule", out/'cm_heur')
    P.plot_cm(cen['cm'], "Centralized LR", out/'cm_central')
    P.plot_cm(res_fl['CM'], "FedAvg ANN-BN", out/'cm_fedann')

    # ROC comparison
    P.plot_roc([cen['prob'], res_fl['Prob'], heur_score],
               ["Centralized LR","FedAvg ANN-BN","Heuristic"],
               te['silver_label'].astype(int).values,
               out/'fig_roc_comparison')

    # Ablations for paper (centralized LR)
    def eval_setting(name, cols):
        r=centralized_lr(tr, te, cols)
        acc=(te['silver_label'].astype(int).values==r['pred']).mean()
        return dict(Setting=name, Features=", ".join(cols),
                    AUC=r['auc'], Accuracy=acc, Precision=r['p'], Recall=r['r'], F1=r['f1'])

    full = feat_all
    rm_ip = ['f_device','f_country','f_asn','f_success']
    rm_su = ['f_device','f_country','f_asn','f_ip_clean']
    rm_both = ['f_device','f_country','f_asn']
    only_direct = ['f_ip_clean','f_success']

    rows=[eval_setting("Full (all 5)", full),
          eval_setting("Removal: - f_ip_clean", rm_ip),
          eval_setting("Removal: - f_success", rm_su),
          eval_setting("Removal: - f_ip_clean & - f_success", rm_both),
          eval_setting("Only direct-rule features", only_direct)]
    pd.DataFrame(rows).round(4).to_csv(out/'Table02_ablation_key_metrics.csv', index=False)

    # Save headline metrics (Table 2)
    def row_from_prob(name, prob, y_true):
        pred=(prob>=0.5).astype(int); cm=confusion_matrix(y_true,pred)
        p,r,f1,_=precision_recall_fscore_support(y_true, pred, average='binary', zero_division=0)
        return dict(Model=name, AUC=roc_auc_score(y_true, prob), Accuracy=(y_true==pred).mean(),
                    Precision=p, Recall=r, F1=f1,
                    TN=cm[0,0], FP=cm[0,1], FN=cm[1,0], TP=cm[1,1])
    table2=pd.DataFrame([
        row_from_prob("Heuristic IP rule", heur_score, te['silver_label'].astype(int).values),
        row_from_prob("Centralized LR", cen['prob'], te['silver_label'].astype(int).values),
        row_from_prob("FedAvg ANN-BN", res_fl['Prob'], te['silver_label'].astype(int).values),
    ]).round(4)
    table2.to_csv(out/'Table02_key_metrics.csv', index=False)

    print("\n== Table 2 (headlines) ==\n", table2)
    print("\nSaved outputs to:", out.resolve())

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv', default='data/rba_fuzzy_labels.csv')
    ap.add_argument('--out', default='outputs')
    ap.add_argument('--device', default='cpu')  # 'cuda' if you have GPU
    args=ap.parse_args()
    main(args.csv, args.out, args.device)
