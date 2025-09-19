# -*- coding: utf-8 -*-
import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

def plot_policy_tradeoffs(T, y, out_pdf_png):
    taus=np.linspace(0.05,0.95,19)
    ch,prec,rec,f1=[],[],[],[]
    from sklearn.metrics import precision_recall_fscore_support
    for t in taus:
        yhat=(T<t).astype(int)
        ch.append(yhat.mean())
        p,r,f,_=precision_recall_fscore_support(y, yhat, average='binary', zero_division=0)
        prec.append(p); rec.append(r); f1.append(f)
    plt.figure(figsize=(8,5))
    plt.plot(taus,ch,label="Challenge rate")
    plt.plot(taus,prec,label="Precision"); plt.plot(taus,rec,label="Recall"); plt.plot(taus,f1,label="F1")
    plt.xlabel("Threshold τ"); plt.ylabel("Rate / Score"); plt.title("Policy trade-offs")
    plt.legend(); plt.tight_layout()
    for ext in out_pdf_png: plt.savefig(f"{out_pdf_png}.{ext}", dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc(prob_list, labels, y_true, out_pdf_png):
    plt.figure(figsize=(7,6))
    for prob, lab in zip(prob_list, labels):
        fpr,tpr,_=roc_curve(y_true, prob)
        auc=roc_auc_score(y_true, prob)
        plt.plot(fpr,tpr,label=f"{lab} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],'--',label="Chance")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC comparison")
    plt.legend(loc="lower right"); plt.tight_layout()
    for ext in out_pdf_png: plt.savefig(f"{out_pdf_png}.{ext}", dpi=300, bbox_inches='tight')
    plt.close()

def plot_cm(cm, title, out_pdf_png, cmap="Blues"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4.5,4))
    ax=plt.gca(); im=ax.imshow(cm, cmap=cmap)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Pred 0","Pred 1"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["True 0","True 1"])
    ax.set_title(title)
    vmax=cm.max()
    for i in range(2):
        for j in range(2):
            color="white" if cm[i,j]>0.6*vmax else "black"
            ax.text(j,i, f"{int(cm[i,j])}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")
    plt.colorbar(im, fraction=0.046, pad=0.04); plt.tight_layout()
    for ext in out_pdf_png: plt.savefig(f"{out_pdf_png}.{ext}", dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curves(history, out_pdf_png):
    rounds=[h['round'] for h in history]
    acc=[h['Accuracy'] for h in history]
    f1 =[h['F1'] for h in history]
    auc=[h['AUC'] for h in history]
    plt.figure(figsize=(8,5))
    plt.plot(rounds, acc, label="Acc")
    plt.plot(rounds, f1,  label="F1")
    plt.plot(rounds, auc, label="AUC")
    plt.xlabel("Round"); plt.ylabel("Score"); plt.title("FL convergence (validation)")
    plt.legend(); plt.tight_layout()
    for ext in out_pdf_png: plt.savefig(f"{out_pdf_png}.{ext}", dpi=300, bbox_inches='tight')
    plt.close()
