import random
from collections import Counter

import nltk
from nltk.corpus import gutenberg


CORPUS_FILES = [
    'carroll-alice.txt',
    'burgess-busterbrown.txt',
    'bryant-stories.txt',
    'austen-persuasion.txt',
    'chesterton-brown.txt',
]


def _ensure_nltk_data():
    for resource, path in [('gutenberg', 'corpora/gutenberg'),
                            ('punkt', 'tokenizers/punkt'),
                            ('punkt_tab', 'tokenizers/punkt_tab')]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


def load_data(seed=42, val_frac=0.2, test_frac=0.1):
    _ensure_nltk_data()

    random.seed(seed)

    raw_text = " ".join(gutenberg.raw(f) for f in CORPUS_FILES)
    sentences = nltk.sent_tokenize(raw_text)
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    tokenized_sentences = [s for s in tokenized_sentences if len(s) >= 2]
    random.shuffle(tokenized_sentences)

    n = len(tokenized_sentences)
    n_val = int(val_frac * n)
    n_test = int(test_frac * n)
    n_train = n - n_val - n_test

    train_sents = tokenized_sentences[:n_train]
    val_sents = tokenized_sentences[n_train:n_train + n_val]
    test_sents = tokenized_sentences[n_train + n_val:]

    # Word vocab comes from train only, so val/test can contain genuinely
    # unseen words -- that's what exercises the char-CNN's open-vocab property.
    word_counts = Counter(word for sent in train_sents for word in sent)
    vocab_words = ['<unk>', '<bos>', '<eos>'] + sorted(word_counts.keys())
    word2idx = {w: i for i, w in enumerate(vocab_words)}
    idx2word = {i: w for w, i in word2idx.items()}

    all_chars = set()
    for sent in tokenized_sentences:
        for word in sent:
            all_chars.update(word)
    # <bos>/<eos> get their own char-level symbols rather than being spelled out
    char_vocab = ['<bow>', '<eow>', '<unk_char>', '<pad_char>', '<bos>', '<eos>'] + sorted(all_chars)
    char2idx = {c: i for i, c in enumerate(char_vocab)}

    return {
        'train_sents': train_sents,
        'val_sents': val_sents,
        'test_sents': test_sents,
        'word2idx': word2idx,
        'idx2word': idx2word,
        'char2idx': char2idx,
    }


if __name__ == '__main__':
    data = load_data()
    print(f"Train / Val / Test sentence counts: "
          f"{len(data['train_sents'])} / {len(data['val_sents'])} / {len(data['test_sents'])}")
    print(f"Total word tokens (train only): {sum(len(s) for s in data['train_sents'])}")
    print(f"Word vocab size: {len(data['word2idx'])}")
    print(f"Char vocab size: {len(data['char2idx'])}")
