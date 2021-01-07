from fastai.vision.all import *

class MultiModel(Module):
    "A multi-head model given an 'encoder' and 'n' output features"
    def __init__(self, encoder, n):
        nf = num_features_model(encoder) * 2
        self.encoder = encoder
        self.num1 = create_head(nf, n[0])
        self.num2 = create_head(nf, n[1])
        self.op = create_head(nf, n[2])
  
    def forward(self, x):
        y = self.encoder(x)
        num1 = self.num1(y)
        num2 = self.num2(y)
        op = self.op(y)
        return [num1, num2, op]
    
class CombinationLoss(Module):
    "Calculate total loss from 3 targets with custom contribution"
    def __init__(self, func=F.cross_entropy, weights=[1, 1, 1]):
        self.func, self.w = func, weights

    def forward(self, xs, *ys, reduction='mean'):
        loss = 0
        for i, w, x, y in zip(range(len(xs)), self.w, xs, ys):
            loss += w * self.func(x, y, reduction=reduction) 
        return loss
    
def num1_acc(inp, num1_targ, num2_targ, op_targ, axis=-1):
    pred, targ = flatten_check(inp[0].argmax(dim=axis), num1_targ)
    return (pred == targ).float().mean()

def num2_acc(inp, num1_targ, num2_targ, op_targ, axis=-1):
    pred, targ = flatten_check(inp[1].argmax(dim=axis), num2_targ)
    return (pred == targ).float().mean()

def op_acc(inp, num1_targ, num2_targ, op_targ, axis=-1):
    pred, targ = flatten_check(inp[2].argmax(dim=axis), op_targ)
    return (pred == targ).float().mean()

def combine_acc(inp, num1_targ, num2_targ, op_targ, axis=-1):
    pred1, targ1 = flatten_check(inp[0].argmax(dim=axis), num1_targ)
    pred2, targ2 = flatten_check(inp[1].argmax(dim=axis), num2_targ)
    pred3, targ3 = flatten_check(inp[2].argmax(dim=axis), op_targ)
    acc1 = pred1 == targ1
    acc2 = pred2 == targ2
    acc3 = pred3 == targ3
    acc = acc1 & acc2 & acc3
    return acc.float().mean()
    
def multimodel_split(m): return L(m.encoder, nn.Sequential(m.num1, m.num2, m.op)).map(params)

@patch
def multimodel_predict(self:Learner, item):
    dl = self.dls.test_dl([item], num_workers=0)
    preds, _ = self.get_preds(dl=dl)
    num1 = self.dls.vocab[0][preds[0].argmax(dim=1)][0]
    num2 = self.dls.vocab[1][preds[1].argmax(dim=1)][0]
    op   = self.dls.vocab[2][preds[2].argmax(dim=1)][0]
    if op == 'plus': 
        ans = num1 + num2
        op = '+'
    if op == 'minus': 
        ans = num1 - num2
        op = '-'
    if op == 'times': 
        ans = num1 * num2
        op = '*'
    if op == 'divide': 
        if num2 == 0: return f'Division by zero!'
        ans = num1 / num2
        op = '/'
        if num1 % num2: return f'{num1} {op} {num2} = {ans:.1f}'
    return f'{num1} {op} {num2} = {ans}'