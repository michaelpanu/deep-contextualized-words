import math
import random
import sys
import time

import torch
import torch.nn as nn

from bilm import BiLM
from char_cnn import encode_sentence
from data import load_data

LR = 1e-3
GRAD_CLIP = 5.0


def sentence_tensors(sent, word2idx, char2idx):
    tokens = ['<bos>'] + sent + ['<eos>']
    char_ids = encode_sentence(tokens, char2idx)
    unk = word2idx['<unk>']
    word_ids = torch.tensor([word2idx.get(w, unk) for w in tokens])
    return char_ids, word_ids


def sentence_losses(model, char_ids, word_ids, criterion):
    fwd_logits, bwd_logits = model(char_ids)
    fwd_loss = criterion(fwd_logits, word_ids[1:])           # predict the next word
    bwd_loss = criterion(bwd_logits, word_ids.flip(0)[1:])   # predict the previous word
    return fwd_loss, bwd_loss


def perplexities(model, sents, word2idx, char2idx, criterion):
    model.eval()
    fwd_total = bwd_total = n = 0.0
    with torch.no_grad():
        for sent in sents:
            char_ids, word_ids = sentence_tensors(sent, word2idx, char2idx)
            fwd_loss, bwd_loss = sentence_losses(model, char_ids, word_ids, criterion)
            t = len(word_ids) - 1
            fwd_total += fwd_loss.item() * t
            bwd_total += bwd_loss.item() * t
            n += t
    model.train()
    return math.exp(fwd_total / n), math.exp(bwd_total / n)


def check_no_leakage(model, sent, char2idx):
    """Each direction must be blind to the tokens it is supposed to predict."""
    model.eval()
    k = len(sent)
    with torch.no_grad():
        base_fwd, base_bwd = model(encode_sentence(['<bos>'] + sent + ['<eos>'], char2idx))
        swapped_tail = sent[:-1] + ['zzzz']
        swapped_head = ['zzzz'] + sent[1:]
        fwd2, _ = model(encode_sentence(['<bos>'] + swapped_tail + ['<eos>'], char2idx))
        _, bwd2 = model(encode_sentence(['<bos>'] + swapped_head + ['<eos>'], char2idx))
    model.train()
    assert torch.allclose(base_fwd[:k], fwd2[:k], atol=1e-5), "forward LM can see future tokens"
    assert torch.allclose(base_bwd[:k], bwd2[:k], atol=1e-5), "backward LM can see past tokens"
    print(f"No-leakage check passed on a {k}-word sentence (both directions).")


if __name__ == '__main__':
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    torch.manual_seed(0)
    random.seed(0)

    data = load_data()
    train_sents, val_sents = data['train_sents'], data['val_sents']
    word2idx, char2idx = data['word2idx'], data['char2idx']

    model = BiLM(len(char2idx), len(word2idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    check_no_leakage(model, train_sents[0], char2idx)

    best_ppl = float('inf')
    for epoch in range(1, epochs + 1):
        random.shuffle(train_sents)
        start = time.time()
        total = n = 0.0
        for sent in train_sents:
            char_ids, word_ids = sentence_tensors(sent, word2idx, char2idx)
            fwd_loss, bwd_loss = sentence_losses(model, char_ids, word_ids, criterion)
            loss = fwd_loss + bwd_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            t = len(word_ids) - 1
            total += loss.item() * t
            n += t
        fwd_ppl, bwd_ppl = perplexities(model, val_sents, word2idx, char2idx, criterion)
        mean_ppl = (fwd_ppl + bwd_ppl) / 2
        marker = ""
        if mean_ppl < best_ppl:
            best_ppl = mean_ppl
            torch.save(model.state_dict(), 'bilm.pt')
            marker = " <- saved"
        print(f"epoch {epoch:2d} | train loss {total / n:.3f} | "
              f"val ppl fwd {fwd_ppl:7.1f} bwd {bwd_ppl:7.1f} | "
              f"{time.time() - start:.0f}s{marker}")

    print(f"best val perplexity {best_ppl:.1f} (bilm.pt)")
