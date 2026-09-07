import itertools
import json
import re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'static only': '#8c8c8c', 'ELMo at input': '#1f77b4', 'ELMo at input+output': '#d62728'}


def parse_training(path, start_epoch=0):
    pattern = re.compile(r'epoch\s+(\d+) \| train loss ([\d.]+) \| '
                         r'val ppl fwd\s+([\d.]+) bwd\s+([\d.]+)')
    rows = []
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                rows.append((int(m.group(1)) + start_epoch, float(m.group(2)),
                             float(m.group(3)), float(m.group(4))))
    return rows


def plot_bilm_training():
    initial = parse_training('train_log.txt')
    resumed = parse_training('train_resume_log.txt', start_epoch=len(initial))
    rows = initial + resumed
    epochs = [r[0] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [r[1] for r in rows], color='#1f77b4')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('training loss (fwd + bwd)')
    ax1.set_title('biLM training loss')

    ax2.plot(epochs, [r[2] for r in rows], label='forward', color='#1f77b4')
    ax2.plot(epochs, [r[3] for r in rows], label='backward', color='#d62728', linestyle='--')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('validation perplexity')
    ax2.set_title('biLM validation perplexity')
    ax2.legend()

    for ax in (ax1, ax2):
        ax.axvline(len(initial) + 0.5, color='gray', linestyle=':', linewidth=1)
        ax.grid(alpha=0.3)
    ax2.annotate('learning rate\ndropped 5x', xy=(len(initial) + 0.7, 210),
                 fontsize=8, color='gray')

    fig.tight_layout()
    fig.savefig('fig1_bilm_training.png', dpi=150)
    print('wrote fig1_bilm_training.png')


def plot_low_resource():
    with open('sweep_results.json') as f:
        rows = json.load(f)

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r['config'], r['sentences'])].append(r['test_acc'])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    configs = sorted({r['config'] for r in rows}, key=lambda c: list(COLORS).index(c))
    for config in configs:
        sizes = sorted({s for c, s in grouped if c == config})
        means = [sum(grouped[(config, s)]) / len(grouped[(config, s)]) for s in sizes]
        lo = [m - min(grouped[(config, s)]) for m, s in zip(means, sizes)]
        hi = [max(grouped[(config, s)]) - m for m, s in zip(means, sizes)]
        ax.errorbar(sizes, means, yerr=[lo, hi], label=config, marker='o',
                    capsize=3, color=COLORS[config])

    ax.set_xscale('log')
    ax.set_xlabel('labeled training sentences (log scale)')
    ax.set_ylabel('POS test accuracy')
    ax.set_title('ELMo helps most when labeled data is scarce')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('fig2_low_resource.png', dpi=150)
    print('wrote fig2_low_resource.png')


def plot_sense_separation():
    from elmo import SENSE_GROUPS, ScalarMix, cosine, layer_vectors, load_bilm

    model, data = load_bilm()
    mixer = ScalarMix(3)
    items = []
    for sense, sentences in SENSE_GROUPS.items():
        for words in sentences:
            layers = layer_vectors(model, words, words.index('well'), data['char2idx'])
            import torch
            with torch.no_grad():
                items.append({'sense': sense, 'static': layers[0], 'layer1': layers[1],
                              'layer2': layers[2], 'elmo': mixer(layers)})

    reps = ['static', 'layer1', 'layer2', 'elmo']
    labels = ['$h_0$\n(char-CNN)', '$h_1$\n(layer 1)', '$h_2$\n(layer 2)', 'ELMo\n(uniform mix)']
    same_means, diff_means = [], []
    for rep in reps:
        same = [cosine(a[rep], b[rep]) for a, b in itertools.combinations(items, 2)
                if a['sense'] == b['sense']]
        diff = [cosine(a[rep], b[rep]) for a, b in itertools.combinations(items, 2)
                if a['sense'] != b['sense']]
        same_means.append(sum(same) / len(same))
        diff_means.append(sum(diff) / len(diff))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(reps))
    ax.bar([i - 0.2 for i in x], same_means, 0.4, label='same sense', color='#1f77b4')
    ax.bar([i + 0.2 for i in x], diff_means, 0.4, label='different sense', color='#d62728')
    for i, (s, d) in enumerate(zip(same_means, diff_means)):
        ax.annotate(f'gap\n{s - d:.3f}', (i, max(s, d) + 0.03), ha='center', fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel('mean cosine similarity')
    ax.set_ylim(0, 1.18)
    ax.set_title('Separating the two senses of "well"')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig('fig3_sense_separation.png', dpi=150)
    print('wrote fig3_sense_separation.png')


def plot_layer_weights():
    text = open('downstream_log.txt').read()
    mix_rows = re.findall(r'(ELMo at \S+)\s+(input|output)\s+s = \[([\d., ]+)\]\s+gamma = ([\d.]+)',
                          text)
    probe_rows = re.findall(r'(h_\d) \([^)]+\)\s+([\d.]+)', text)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    dual = [r for r in mix_rows if r[0] == 'ELMo at input+output']
    width = 0.35
    for i, (_, slot, s_str, gamma) in enumerate(dual):
        weights = [float(v) for v in s_str.split(',')]
        ax1.bar([j + (i - 0.5) * width for j in range(3)], weights, width,
                label=f'{slot} (gamma={float(gamma):.2f})')
    ax1.axhline(1 / 3, color='gray', linestyle=':', label='uniform init (1/3)')
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(['$h_0$', '$h_1$', '$h_2$'])
    ax1.set_ylabel('learned softmax weight $s_j$')
    ax1.set_title('Eq. (1) weights differ by injection point')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, axis='y')

    names = [r[0] for r in probe_rows]
    accs = [float(r[1]) for r in probe_rows]
    ax2.bar(names, accs, color=['#8c8c8c', '#1f77b4', '#1f77b4'])
    for i, a in enumerate(accs):
        ax2.annotate(f'{a:.4f}', (i, a + 0.002), ha='center', fontsize=9)
    ax2.set_ylim(0.85, 0.96)
    ax2.set_ylabel('POS test accuracy')
    ax2.set_title('Linear probe on each frozen layer')
    ax2.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig('fig4_layer_weights.png', dpi=150)
    print('wrote fig4_layer_weights.png')


if __name__ == '__main__':
    plot_bilm_training()
    plot_sense_separation()
    plot_layer_weights()
    plot_low_resource()
