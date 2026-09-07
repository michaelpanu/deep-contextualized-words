import torch
import torch.nn as nn

from char_cnn import CharCNN, CHAR_CNN_DIM

HIDDEN_DIM = 128
NUM_LAYERS = 2


class BiLM(nn.Module):
    """Sec 3.1: shared char-CNN (Theta_x) and softmax (Theta_s), separate LSTM weights per direction."""

    def __init__(self, char_vocab_size, word_vocab_size):
        super().__init__()
        self.char_cnn = CharCNN(char_vocab_size)
        self.fwd_lstms = nn.ModuleList(
            [nn.LSTM(CHAR_CNN_DIM if i == 0 else HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
             for i in range(NUM_LAYERS)])
        self.bwd_lstms = nn.ModuleList(
            [nn.LSTM(CHAR_CNN_DIM if i == 0 else HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
             for i in range(NUM_LAYERS)])
        self.softmax = nn.Linear(HIDDEN_DIM, word_vocab_size)

    def _run_stack(self, token_vecs, lstms):
        h = token_vecs.unsqueeze(0)
        outputs = []
        for i, lstm in enumerate(lstms):
            out, _ = lstm(h)
            if i > 0:
                out = out + h  # residual from layer 1 into layer 2
            outputs.append(out.squeeze(0))
            h = out
        return outputs

    def forward(self, char_ids):
        """char_ids for <bos> w_1..w_N <eos>; returns next-word and previous-word logits."""
        token_vecs = self.char_cnn(char_ids)
        fwd_top = self._run_stack(token_vecs[:-1], self.fwd_lstms)[-1]
        bwd_top = self._run_stack(token_vecs.flip(0)[:-1], self.bwd_lstms)[-1]
        return self.softmax(fwd_top), self.softmax(bwd_top)

    def representations(self, char_ids):
        """R_k as L+1 vectors per token, each 2*HIDDEN_DIM, aligned to input positions."""
        token_vecs = self.char_cnn(char_ids)
        fwd = self._run_stack(token_vecs, self.fwd_lstms)
        bwd = [layer.flip(0) for layer in self._run_stack(token_vecs.flip(0), self.bwd_lstms)]
        layers = [torch.cat([token_vecs, token_vecs], dim=-1)]  # h_0: no direction, so duplicated
        layers += [torch.cat([f, b], dim=-1) for f, b in zip(fwd, bwd)]
        return layers


if __name__ == '__main__':
    from char_cnn import encode_sentence
    from data import load_data

    data = load_data()
    model = BiLM(len(data['char2idx']), len(data['word2idx']))

    sent = data['train_sents'][0]
    tokens = ['<bos>'] + sent + ['<eos>']
    char_ids = encode_sentence(tokens, data['char2idx'])

    fwd_logits, bwd_logits = model(char_ids)
    print(f"Sentence ({len(sent)} words): {sent}")
    print(f"Forward logits:  {tuple(fwd_logits.shape)}  (predictions x vocab)")
    print(f"Backward logits: {tuple(bwd_logits.shape)}")

    layers = model.representations(encode_sentence(sent, data['char2idx']))
    print(f"\nR_k: {len(layers)} layers, each {tuple(layers[0].shape)} (tokens x dim)")
