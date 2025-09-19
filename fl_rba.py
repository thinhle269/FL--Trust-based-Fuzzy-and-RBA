 
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class ANNBN(nn.Module):
    """Compact MLP with BatchNorm: 16-8-BN → sigmoid"""
    def __init__(self, in_dim=5):
        super().__init__()
        self.fc1=nn.Linear(in_dim,16); self.bn1=nn.BatchNorm1d(16)
        self.fc2=nn.Linear(16,8);      self.bn2=nn.BatchNorm1d(8)
        self.out=nn.Linear(8,1)
        self.act=nn.ReLU()

    def forward(self,x):
        x=self.act(self.bn1(self.fc1(x)))
        x=self.act(self.bn2(self.fc2(x)))
        return self.out(x)  # logits

def to_device(*tensors, device='cpu'):
    return [torch.tensor(t, dtype=torch.float32, device=device) for t in tensors]

class Client:
    def __init__(self, X, y, batch=32, device='cpu'):
        X, y = to_device(X, y, device=device)
        self.loader=DataLoader(TensorDataset(X,y), batch_size=batch, shuffle=True)
        self.device=device

    def local_train(self, global_model, epochs=1, lr=1e-2, class_weight=None):
        model=ANNBN(in_dim=global_model.fc1.in_features).to(self.device)
        model.load_state_dict(global_model.state_dict())
        criterion=nn.BCEWithLogitsLoss(pos_weight=None if class_weight is None else torch.tensor([class_weight], device=self.device))
        opt=optim.SGD(model.parameters(), lr=lr, momentum=0.0)
        model.train()
        for _ in range(epochs):
            for xb,yb in self.loader:
                opt.zero_grad()
                logits=model(xb).squeeze(1)
                loss=criterion(logits, yb)
                loss.backward(); opt.step()
        return model.state_dict()

class Server:
    def __init__(self, in_dim=5, device='cpu'):
        self.model=ANNBN(in_dim=in_dim).to(device)
        self.device=device

    def aggregate(self, states, sizes):
        """FedAvg: weighted by client data size"""
        new_sd={k: torch.zeros_like(v) for k,v in states[0].items()}
        total=float(sum(sizes))
        for sd, n in zip(states, sizes):
            w=float(n)/total
            for k in new_sd: new_sd[k]+=sd[k]*w
        self.model.load_state_dict(new_sd)

    @torch.no_grad()
    def evaluate(self, X, y):
        self.model.eval()
        X, y = to_device(X, y, device=self.device)
        logits=self.model(X).squeeze(1)
        prob=torch.sigmoid(logits).cpu().numpy()
        yhat=(prob>=0.5).astype(int)
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
        acc=accuracy_score(y, yhat)
        p,r,f1,_=precision_recall_fscore_support(y, yhat, average='binary', zero_division=0)
        try: auc=roc_auc_score(y, prob)
        except: auc=np.nan
        cm=confusion_matrix(y, yhat)
        return dict(Accuracy=acc, Precision=p, Recall=r, F1=f1, AUC=auc, CM=cm, Prob=prob, Pred=yhat)

    def run_fedavg(self, clients, rounds=50, local_epochs=1, lr=1e-2, class_weight=None, val_data=None, log_every=5):
        history=[]
        for r in range(1, rounds+1):
            states=[]; sizes=[]
            for c in clients:
                sd=c.local_train(self.model, epochs=local_epochs, lr=lr, class_weight=class_weight)
                states.append({k:v.detach().clone() for k,v in sd.items()})
                sizes.append(len(c.loader.dataset))
            self.aggregate(states, sizes)
            if val_data and (r==1 or r%log_every==0 or r==rounds):
                Xv,Yv=val_data
                met=self.evaluate(Xv,Yv); met['round']=r; history.append(met)
                print(f"[Round {r}] Acc={met['Accuracy']:.4f} F1={met['F1']:.4f} AUC={met['AUC']:.4f}")
        return history
