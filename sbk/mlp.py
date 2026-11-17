"""Multi-task softmax networks in numpy: one shared trunk over the one-hot
GIVEN block, one softmax head per PREDICT item (segment softmax over a flat
logit vector). hidden=() gives a joint (per-item) multinomial logistic model.

Cross-fitting: one model per fold trained on the other folds with an inner
early-stopping split; OOF = that model on its fold; test = mean of the fold
models on the hidden rows.
"""
import time

import numpy as np


def onehot(X, cards):
    offs = np.concatenate([[0], np.cumsum(cards)[:-1]])
    D = int(cards.sum())
    out = np.zeros((X.shape[0], D), dtype=np.float32)
    rows = np.arange(X.shape[0])
    for j in range(X.shape[1]):
        out[rows, offs[j] + X[:, j]] = 1.0
    return out


def pair_features(X, cards, pairs, max_card=400):
    """One-hot of pairwise interactions for the given column pairs."""
    cols, cc = [], []
    for a, b in pairs:
        if cards[a] * cards[b] > max_card:
            continue
        cols.append(X[:, a] * cards[b] + X[:, b])
        cc.append(cards[a] * cards[b])
    if not cols:
        return np.zeros((X.shape[0], 0), np.float32)
    return onehot(np.stack(cols, 1), np.asarray(cc))


class Seg:
    def __init__(self, K):
        self.K = np.asarray(K, dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.K)[:-1]])
        self.S = int(self.K.sum())
        self.T = len(self.K)

    def softmax(self, Z):
        m = np.maximum.reduceat(Z, self.starts, axis=1)
        E = np.exp(Z - np.repeat(m, self.K, axis=1))
        den = np.add.reduceat(E, self.starts, axis=1)
        return E / np.repeat(den, self.K, axis=1)

    def target_index(self, Y):
        return Y + self.starts[None, :]


def _init_bias(Y, seg, smooth=0.5):
    b = np.zeros(seg.S, np.float32)
    for t in range(seg.T):
        c = np.bincount(Y[:, t], minlength=seg.K[t]) + smooth
        b[seg.starts[t]:seg.starts[t] + seg.K[t]] = np.log(c / c.sum())
    return b


class Net:
    def __init__(self, D, hidden, S, rng, bias):
        self.layers = []
        prev = D
        for h in hidden:
            W = (rng.standard_normal((prev, h)) * np.sqrt(2.0 / prev)).astype(np.float32)
            self.layers.append([W, np.zeros(h, np.float32)])
            prev = h
        Wo = (rng.standard_normal((prev, S)) * (0.01 if hidden else 0.0)).astype(np.float32)
        self.layers.append([Wo, bias.astype(np.float32).copy()])
        self.m = [[np.zeros_like(p) for p in l] for l in self.layers]
        self.v = [[np.zeros_like(p) for p in l] for l in self.layers]
        self.t = 0

    def forward(self, A, drop=0.0, rng=None):
        acts, masks = [A], []
        h = A
        for i, (W, b) in enumerate(self.layers[:-1]):
            h = h @ W + b
            h = np.maximum(h, 0.0)
            if drop > 0 and rng is not None:
                mk = (rng.random(h.shape) >= drop).astype(np.float32) / (1 - drop)
                h = h * mk
                masks.append(mk)
            else:
                masks.append(None)
            acts.append(h)
        W, b = self.layers[-1]
        return h @ W + b, acts, masks

    def step(self, grads, lr, wd, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for li, (layer, g_l) in enumerate(zip(self.layers, grads)):
            for pi, (p, g) in enumerate(zip(layer, g_l)):
                if pi == 0 and wd > 0:
                    g = g + wd * p
                m = self.m[li][pi] = b1 * self.m[li][pi] + (1 - b1) * g
                v = self.v[li][pi] = b2 * self.v[li][pi] + (1 - b2) * g * g
                mh = m / (1 - b1 ** self.t)
                vh = v / (1 - b2 ** self.t)
                layer[pi] -= lr * mh / (np.sqrt(vh) + eps)


def _loss_grad(net, A, Y, seg, drop, rng, item_w):
    Z, acts, masks = net.forward(A, drop, rng)
    P = seg.softmax(Z)
    B = A.shape[0]
    idx = seg.target_index(Y)
    rows = np.arange(B)[:, None]
    lp = np.log(np.maximum(P[rows, idx], 1e-12))          # (B, T)
    loss = -float((lp * item_w[None, :]).sum() / B)
    G = P.copy()
    G[rows, idx] -= 1.0
    G *= np.repeat(item_w, seg.K)[None, :] / B
    grads = []
    # Output layer
    h = acts[-1]
    grads.append([h.T @ G, G.sum(0)])
    dh = G @ net.layers[-1][0].T
    for li in range(len(net.layers) - 2, -1, -1):
        if masks[li] is not None:
            dh = dh * masks[li]
        dh = dh * (acts[li + 1] > 0)
        W, b = net.layers[li]
        grads.append([acts[li].T @ dh, dh.sum(0)])
        if li > 0:
            dh = dh @ W.T
    grads.reverse()
    return loss, grads


def predict_net(net, A, seg, bs=4096):
    out = []
    for i in range(0, A.shape[0], bs):
        Z, _, _ = net.forward(A[i:i + bs])
        out.append(seg.softmax(Z))
    return np.concatenate(out, 0)


def eval_loss(net, A, Y, seg, item_w):
    P = predict_net(net, A, seg)
    idx = seg.target_index(Y)
    lp = np.log(np.maximum(P[np.arange(A.shape[0])[:, None], idx], 1e-12))
    return -float((lp * item_w[None, :]).sum() / A.shape[0])


def train_net(A, Y, seg, hidden, rng, epochs=60, bs=256, lr=3e-3, wd=1e-5,
              drop=0.1, patience=6, val_frac=0.12, deadline=None,
              item_w=None):
    n = A.shape[0]
    item_w = (np.full(seg.T, 1.0 / seg.T, np.float32) if item_w is None
              else item_w)
    perm = rng.permutation(n)
    nv = max(1, int(val_frac * n))
    va, tr = perm[:nv], perm[nv:]
    net = Net(A.shape[1], hidden, seg.S, rng, _init_bias(Y[tr], seg))
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        order = tr[rng.permutation(tr.size)]
        for i in range(0, order.size, bs):
            b = order[i:i + bs]
            _, g = _loss_grad(net, A[b], Y[b], seg, drop, rng, item_w)
            net.step(g, lr, wd)
        vl = eval_loss(net, A[va], Y[va], seg, item_w)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = [[p.copy() for p in l] for l in net.layers]
        else:
            bad += 1
            if bad >= patience:
                break
        if deadline and time.time() > deadline:
            break
    if best_state is not None:
        net.layers = best_state
    return net


def mlp_learner(inst, hidden=(128,), pairs=None, epochs=60, bs=256, lr=3e-3,
                wd=1e-5, drop=0.1, seeds=(0,), patience=6, deadline=None):
    A = onehot(inst.X, inst.cards)
    if pairs:
        A = np.concatenate([A, pair_features(inst.X, inst.cards, pairs)], 1)
    AF, AH = A[inst.F], A[inst.H]
    Y = inst.Y
    seg = Seg(inst.K)
    oof = np.zeros((AF.shape[0], seg.S), np.float32)
    test = np.zeros((AH.shape[0], seg.S), np.float32)
    nfits = 0
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        for f in range(inst.n_folds):
            tr = np.flatnonzero(inst.folds != f)
            va = np.flatnonzero(inst.folds == f)
            net = train_net(AF[tr], Y[tr], seg, hidden, rng, epochs, bs, lr,
                            wd, drop, patience, deadline=deadline)
            oof[va] += predict_net(net, AF[va], seg) / len(seeds)
            test += predict_net(net, AH, seg)
            nfits += 1
    test /= nfits
    oofs = [oof[:, s:s + k] for s, k in zip(seg.starts, seg.K)]
    tests = [test[:, s:s + k] for s, k in zip(seg.starts, seg.K)]
    return oofs, tests
