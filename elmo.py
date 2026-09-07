import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from bilm import BiLM
from char_cnn import encode_sentence
from data import load_data


class ScalarMix(nn.Module):
    """Eq. (1): gamma * sum_j softmax(s)_j * h_kj. At init this is a plain average with gamma=1."""

    def __init__(self, num_layers):
        super().__init__()
        self.s = nn.Parameter(torch.zeros(num_layers))
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, layers):
        stacked = layers if torch.is_tensor(layers) else torch.stack(layers, dim=0)
        weights = torch.softmax(self.s, dim=0).view(-1, *([1] * (stacked.dim() - 1)))
        return self.gamma * (weights * stacked).sum(dim=0)


def load_bilm(path='bilm.pt'):
    data = load_data()
    model = BiLM(len(data['char2idx']), len(data['word2idx']))
    model.load_state_dict(torch.load(path))
    model.eval()
    return model, data


def layer_vectors(model, words, target_index, char2idx):
    """R_k at one token position, with <bos>/<eos> present so context matches training."""
    tokens = ['<bos>'] + words + ['<eos>']
    with torch.no_grad():
        layers = model.representations(encode_sentence(tokens, char2idx))
    return [layer[target_index + 1] for layer in layers]


def cosine(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


SENSE_GROUPS = {
    'well (noun: a deep hole)': [
        "she fell down the deep well into the dark".split(),
        "the well was very deep and very dark".split(),
        "down down down into the well she went".split(),
    ],
    'well (adverb: skilfully)': [
        "she did not know the way very well".split(),
        "he could not do it very well at all".split(),
        "you do not play the game very well".split(),
    ],
}


def sense_experiment(model, char2idx):
    mixer = ScalarMix(num_layers=3)  # untrained: uniform weights, gamma = 1

    items = []
    for sense, sentences in SENSE_GROUPS.items():
        for words in sentences:
            idx = words.index('well')
            layers = layer_vectors(model, words, idx, char2idx)
            with torch.no_grad():
                elmo = mixer(layers)
            items.append({
                'sense': sense,
                'text': ' '.join(words),
                'static': layers[0],   # h_0: char-CNN only, no context
                'layer1': layers[1],
                'layer2': layers[2],
                'elmo': elmo,
            })

    reps = ['static', 'layer1', 'layer2', 'elmo']
    print(f"Comparing {len(items)} sentences containing 'well' "
          f"({len(SENSE_GROUPS['well (noun: a deep hole)'])} noun, "
          f"{len(SENSE_GROUPS['well (adverb: skilfully)'])} adverb)\n")
    print(f"{'representation':<14} {'same sense':>11} {'diff sense':>11} {'gap':>8}")
    for rep in reps:
        same, diff = [], []
        for a, b in itertools.combinations(items, 2):
            score = cosine(a[rep], b[rep])
            (same if a['sense'] == b['sense'] else diff).append(score)
        same_mean = sum(same) / len(same)
        diff_mean = sum(diff) / len(diff)
        print(f"{rep:<14} {same_mean:>11.4f} {diff_mean:>11.4f} {same_mean - diff_mean:>8.4f}")

    print("\nPairwise cosine, ELMo vs static:")
    for a, b in itertools.combinations(items, 2):
        tag = "same" if a['sense'] == b['sense'] else "DIFF"
        print(f"  [{tag}] elmo {cosine(a['elmo'], b['elmo']):.3f} | "
              f"static {cosine(a['static'], b['static']):.3f}  "
              f"| {a['text'][:34]:<34} || {b['text'][:34]}")


if __name__ == '__main__':
    model, data = load_bilm()
    sense_experiment(model, data['char2idx'])
