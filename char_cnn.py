import torch
import torch.nn as nn
import torch.nn.functional as F

KERNEL_WIDTHS = (2, 3, 4, 5)
FILTERS_PER_WIDTH = 32  # 4 widths x 32 filters = 128-dim word vector
CHAR_EMBED_DIM = 16
CHAR_CNN_DIM = len(KERNEL_WIDTHS) * FILTERS_PER_WIDTH
# Fixed width, so a word's char-CNN vector never depends on the sentence it sits in.
MAX_WORD_LEN = 20


class CharCNN(nn.Module):
    def __init__(self, char_vocab_size):
        super().__init__()
        self.char_embed = nn.Embedding(char_vocab_size, CHAR_EMBED_DIM)
        self.convs = nn.ModuleList([
            nn.Conv1d(CHAR_EMBED_DIM, FILTERS_PER_WIDTH, kernel_size=w)
            for w in KERNEL_WIDTHS
        ])

    def forward(self, char_ids):
        # char_ids: (num_words, max_word_len) -> (num_words, 128)
        x = self.char_embed(char_ids).transpose(1, 2)  # (num_words, embed_dim, max_word_len)
        pooled = [F.relu(conv(x)).max(dim=-1).values for conv in self.convs]  # max-over-time
        return torch.cat(pooled, dim=-1)


def word_to_char_ids(word, char2idx):
    symbols = [word] if word in ('<bos>', '<eos>') else list(word)[:MAX_WORD_LEN - 2]
    chars = ['<bow>'] + symbols + ['<eow>']
    ids = [char2idx.get(c, char2idx['<unk_char>']) for c in chars]
    return ids + [char2idx['<pad_char>']] * (MAX_WORD_LEN - len(ids))


def encode_sentence(words, char2idx):
    return torch.tensor([word_to_char_ids(w, char2idx) for w in words], dtype=torch.long)


if __name__ == '__main__':
    from data import load_data

    data = load_data()
    char2idx = data['char2idx']

    model = CharCNN(char_vocab_size=len(char2idx))

    sentence = data['train_sents'][0]
    char_ids = encode_sentence(sentence, char2idx)
    word_vecs = model(char_ids)

    print(f"Sentence: {sentence}")
    print(f"Char id tensor shape: {tuple(char_ids.shape)}")
    print(f"Word vector shape: {tuple(word_vecs.shape)}")

    # a made-up, definitely-unseen "word" should still produce a normal-looking vector
    fake_word = "zqxvblorp"
    fake_ids = encode_sentence([fake_word], char2idx)
    fake_vec = model(fake_ids)
    print(f"\nUnseen word '{fake_word}' -> vector shape {tuple(fake_vec.shape)}, "
          f"norm {fake_vec.norm().item():.3f}")
