import json
import random

import torch
import torch.nn as nn

from downstream import Tagger, accuracy, cache_features, tag_labels
from elmo import load_bilm

FRACTIONS = [0.01, 0.05, 0.25, 1.0]
SEEDS = [0, 1, 2]
CONFIGS = [
    ('static only', False, False),
    ('ELMo at input', True, False),
    ('ELMo at input+output', True, True),
]
VAL_SUBSET = 600  # model selection only; final numbers always use the full test set
LR = 1e-3
TAGS = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', 'X', '.']


def epochs_for(fraction):
    # smaller subsets get more passes so every setting gets a fair shot at converging
    return min(30, max(6, round(6 / fraction)))


def train_model(model, feats, labels, val_feats, val_labels, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    order = list(range(len(feats)))
    best_acc, best_state = -1.0, None
    for _ in range(epochs):
        random.shuffle(order)
        for i in order:
            loss = criterion(model(feats[i]), labels[i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = accuracy(model, val_feats, val_labels)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)


if __name__ == '__main__':
    tag2idx = {t: i for i, t in enumerate(TAGS)}
    bilm, data = load_bilm()
    char2idx = data['char2idx']

    print("caching frozen biLM features...")
    feats, labels = {}, {}
    for split in ['train', 'val', 'test']:
        sents = data[f'{split}_sents']
        feats[split] = cache_features(bilm, sents, char2idx)
        labels[split] = tag_labels(sents, tag2idx)

    val_feats = feats['val'][:VAL_SUBSET]
    val_labels = labels['val'][:VAL_SUBSET]
    n_train = len(feats['train'])

    results = []
    for fraction in FRACTIONS:
        n = max(20, int(n_train * fraction))
        epochs = epochs_for(fraction)
        for seed in SEEDS:
            random.seed(seed)
            subset = random.sample(range(n_train), n)
            sub_feats = [feats['train'][i] for i in subset]
            sub_labels = [labels['train'][i] for i in subset]

            for name, use_in, use_out in CONFIGS:
                torch.manual_seed(seed)
                random.seed(seed)
                tagger = Tagger(len(TAGS), use_in, use_out)
                train_model(tagger, sub_feats, sub_labels, val_feats, val_labels, epochs)
                test_acc = accuracy(tagger, feats['test'], labels['test'])

                record = {'fraction': fraction, 'sentences': n, 'seed': seed,
                          'config': name, 'epochs': epochs, 'test_acc': test_acc}
                for slot, mix in [('input', tagger.in_mix), ('output', tagger.out_mix)]:
                    if mix is not None:
                        record[f's_{slot}'] = torch.softmax(mix.s, dim=0).tolist()
                        record[f'gamma_{slot}'] = mix.gamma.item()
                results.append(record)
                print(f"frac {fraction:<5} ({n:5d} sents) seed {seed} "
                      f"{name:<22} test acc {test_acc:.4f}")

            with open('sweep_results.json', 'w') as f:
                json.dump(results, f, indent=2)

    print("\nwrote sweep_results.json")
