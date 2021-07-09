from fastai.vision.all import *
import streamlit as st
import time
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
        if num1 % num2: return num1, num2, op, f'{num1} {op} {num2} = {ans:.3f}'
        else: ans = (int) (num1 / num2)
    return num1, num2, op, f'{num1} {op} {num2} = {ans}'

def predict(img):
    with st.spinner('Wait for it...'): time.sleep(3)
    learn = load_learner('models/number.pkl')
    num1, num2, op, pred = learn.multimodel_predict(img)
    st.success(f"The answer to your expression is: {pred} 😼")
    with st.spinner('Wait for a surprise 😸 ...'): time.sleep(3)
    illustrate(num1.item(), num2.item(), op)

def illustrate(num1, num2, op):
    # not working for non-positive values
    if num1 <= 0 or num2 <= 0: 
        st.error('Oops, no surprise 😿! Better luck next time 😹')
        return

    animation = 'animations/shark.png'
    animation2 = 'animations/shark_net.png'
    limit = 50

    if op == '+':
        st.info('An animation to illustrate this addition equation 🙀')
        cnt = 0
        num1_cols = st.beta_columns(num1)
        for col in num1_cols:
            time.sleep(0.2)
            cnt += 1 
            col.image(animation, width=limit)
            col.title(f'{cnt}')
        time.sleep(0.5)
        num2_cols = st.beta_columns(num2)
        for col in num2_cols: 
            time.sleep(0.2)
            cnt += 1 
            col.image(animation, width=limit)
            col.title(f'{cnt}')
        st.success('Easy to understand now? 😻')
        return

    if op == '-' and num1 - num2 >= 0:
        st.info('An animation to illustrate this subtraction equation 🙀')
        cnt = 0
        num1_cols = st.beta_columns(num1)
        for col in num1_cols:
            time.sleep(0.2)
            cnt += 1 
            col.image(animation, width=limit)
            col.title(f'{cnt}')
        time.sleep(0.5)
        num2_cols = st.beta_columns(num2)
        for col in num2_cols: 
            time.sleep(0.2)
            cnt -= 1 
            col.image(animation2, width=limit)
            col.title(f'{cnt}')
        st.success('Easy to understand now? 😻')
        return

    if op == '*':
        st.info('An animation to illustrate this multiplication equation 🙀')
        cnt = 0
        cols = st.beta_columns(num1)
        for i in range(num2):
            for col in cols:
                time.sleep(0.1)
                cnt += 1 
                col.image(animation, width=limit)
                col.title(f'{cnt}')
            time.sleep(0.3)
        st.success('Easy to understand now? 😻')
        return
    
    if op == '/' and num1 % num2 == 0:
        st.info('An animation to illustrate this division equation 🙀')
        cnt = 0
        cols = st.beta_columns(num2)
        for i in range((int) (num1 / num2)):
            for col in cols:
                time.sleep(0.2)
                cnt += 1 
                col.image(animation, width=limit)
                col.title(f'{cnt}')    
            time.sleep(0.5)   
        st.success('Easy to understand now? 😻')
        return 
    
    st.error('Oops, no surprise 😿! Better luck next time 😹')