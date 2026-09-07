import random

import nltk
import torch
import torch.nn as nn

from char_cnn import encode_sentence
from elmo import ScalarMix, load_bilm

REP_DIM = 256
TASK_HIDDEN = 64
EPOCHS = 8
LR = 1e-3


class Tagger(nn.Module):
    """Sec 3.3 Pipeline B: task biLSTM + output layer, optionally fed [x_k ; ELMo_k]."""

    def __init__(self, num_tags, use_elmo, output_elmo=False):
        super().__init__()
        self.in_mix = ScalarMix(3) if use_elmo else None
        self.out_mix = ScalarMix(3) if output_elmo else None  # own s/gamma, learned separately
        in_dim = REP_DIM * 2 if use_elmo else REP_DIM
        out_dim = TASK_HIDDEN * 2 + (REP_DIM if output_elmo else 0)
        self.rnn = nn.LSTM(in_dim, TASK_HIDDEN, batch_first=True, bidirectional=True)
        self.out = nn.Linear(out_dim, num_tags)

    def forward(self, layers):
        x = layers[0]  # x_k: the context-independent token layer
        if self.in_mix is not None:
            x = torch.cat([x, self.in_mix(layers)], dim=-1)
        hidden, _ = self.rnn(x.unsqueeze(0))
        h = hidden.squeeze(0)
        if self.out_mix is not None:
            h = torch.cat([h, self.out_mix(layers)], dim=-1)
        return self.out(h)


class LinearProbe(nn.Module):
    """Sec 5.3-style probe: linear classifier straight off one frozen biLM layer."""

    def __init__(self, num_tags, layer_index):
        super().__init__()
        self.layer_index = layer_index
        self.out = nn.Linear(REP_DIM, num_tags)

    def forward(self, layers):
        return self.out(layers[self.layer_index])


def cache_features(model, sents, char2idx):
    """Run the frozen biLM once per sentence; these features never change during task training."""
    cached = []
    with torch.no_grad():
        for sent in sents:
            tokens = ['<bos>'] + sent + ['<eos>']
            layers = model.representations(encode_sentence(tokens, char2idx))
            cached.append(torch.stack([layer[1:-1] for layer in layers], dim=0))
    return cached


def tag_labels(sents, tag2idx):
    return [torch.tensor([tag2idx[t] for _, t in nltk.pos_tag(s, tagset='universal')]) for s in sents]


def accuracy(model, features, labels):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for feats, gold in zip(features, labels):
            pred = model(feats).argmax(dim=-1)
            correct += (pred == gold).sum().item()
            total += len(gold)
    model.train()
    return correct / total


def train_task_model(model, train_feats, train_labels, val_feats, val_labels, name):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    order = list(range(len(train_feats)))
    best_acc, best_state = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        random.shuffle(order)
        for i in order:
            loss = criterion(model(train_feats[i]), train_labels[i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = accuracy(model, val_feats, val_labels)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  {name} epoch {epoch}: val acc {acc:.4f}")

    model.load_state_dict(best_state)
    return best_acc


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    model, data = load_bilm()
    char2idx = data['char2idx']

    tags = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', 'X', '.']
    tag2idx = {t: i for i, t in enumerate(tags)}

    splits = {}
    for name in ['train', 'val', 'test']:
        sents = data[f'{name}_sents']
        splits[name] = (cache_features(model, sents, char2idx), tag_labels(sents, tag2idx))
    print(f"Cached frozen biLM features: "
          f"{', '.join(f'{k} {len(v[0])} sents' for k, v in splits.items())}\n")

    configs = [
        ('static only', False, False),
        ('ELMo at input', True, False),
        ('ELMo at input+output', True, True),
    ]
    results, mixes = {}, {}
    for name, use_elmo, output_elmo in configs:
        torch.manual_seed(0)
        tagger = Tagger(len(tags), use_elmo, output_elmo)
        print(f"Training tagger ({name}):")
        train_task_model(tagger, *splits['train'], *splits['val'], name)
        results[name] = accuracy(tagger, *splits['test'])
        for slot, mix in [('input', tagger.in_mix), ('output', tagger.out_mix)]:
            if mix is not None:
                mixes[(name, slot)] = (torch.softmax(mix.s, dim=0).tolist(), mix.gamma.item())
        print()

    print("Probing each frozen layer with a linear classifier (Sec 5.3 style):")
    probe_acc = {}
    for j, label in enumerate(['h_0 (char-CNN)', 'h_1 (biLSTM layer 1)', 'h_2 (biLSTM layer 2)']):
        torch.manual_seed(0)
        probe = LinearProbe(len(tags), j)
        train_task_model(probe, *splits['train'], *splits['val'], label)
        probe_acc[label] = accuracy(probe, *splits['test'])
        print()

    print("=" * 58)
    print("POS tagging test accuracy")
    for name, _, _ in configs:
        delta = results[name] - results['static only']
        print(f"  {name:<22} {results[name]:.4f}  ({delta:+.4f})")

    print("\nLearned Eq. (1) weights per injection point:")
    for (name, slot), (s, gamma) in mixes.items():
        print(f"  {name:<22} {slot:<7} s = {[round(w, 3) for w in s]}  gamma = {gamma:.3f}")
    print("\nLinear probe test accuracy per frozen layer:")
    for label, acc in probe_acc.items():
        print(f"  {label:<22} {acc:.4f}")
