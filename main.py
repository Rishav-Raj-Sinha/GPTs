import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_blocks import Block

print("Torch version:", torch.__version__)
print("cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

lynch_quotes = [
    "Ideas are like fish. If you want to catch little fish, you can stay in the shallow water. But if you want to catch the big fish, you've got to go deeper.",
    "I don't know why people expect art to make sense when they accept the fact that life doesn't make sense.",
    "We think we understand the rules when we become adults but what we really experience is a narrowing of the imagination.",
    "There's a safety in thinking in a diner. You can have your coffee or your milkshake, and you can go off into strange dark areas, and always come back to the safety of the diner.",
    "Negativity is the enemy of creativity.",
    "I like to remember things my own way. How I remembered them, not necessarily the way they happened.",
    "Intuition is the key to everything, in painting, filmmaking, business — everything.",
    "I learned that just beneath the surface there's another world, and still different worlds as you dig deeper.",
    "This whole world is wild at heart and weird on top.",
    "Inside, we are ageless and when we talk to ourselves, it's the same age of the person we were talking to when we were little. It's the body that is changing around that ageless center.",
]

# joininh the sentences present in the list and adding the end tokens after each sentence
lynch_quotes = [s + " <END>" for s in lynch_quotes]
text = " ".join(lynch_quotes)
# print(text)

# splitting the text into words and removing duplicates using python set
words = list(set(text.split()))
vocab_size = len(words)
# output: 137
# the VOCABULARY for our LLM will be of 137 words
# vocabulary means the set of unique words that the model will use to predict the next word
# anything the model predicts will be a word from this vocabulary

# the next step is to generate vector representations of the words in the vocabulary
# this is done by a tokenizer
# but for the sake of keeping things simple, using a simple approach of giving each token/word a unique ID

word2index = {w: i for i, w in enumerate(words)}
# print("word2index:", word2index)

idx2word = {i: w for i, w in enumerate(words)}

# now we can't pass this dictionary directly to the model
# so we create a tensor
# a tensor is a multi-dimensional array that can be used to represent a wide variety of data

data = torch.tensor([word2index[w] for w in text.split()], dtype=torch.long)
# print(data)

# Batch function
# takes a tensor and splits it into batches of a given size

block_size = 6  # the number of words in each batch, the number of words the model will process at once, it is also the number of words it will see during generating the new word after the training
embedding_dim = 32  # for each word there will be a 32-dimensional vector representation, as the traning goes on the words that are similar will have similar vector representations
n_heads = 2  # the number of attention heads in the transformer model
n_layers = 2  # the number of transformer layers in the model
lr = 1e-3  # the learning rate for the optimizer
epochs = 1500  # the number of epochs to train the model


def get_batch(batch_size=16):
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


class LynchGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(
            *[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape  # B is batch size , T is sequence length
        tok_emb = self.token_embedding(idx)

        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


model = LynchGPT()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=lr
)  # after every epoch the weights will be updated

for step in range(epochs):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad()  # clear gradients from last epoch
    loss.backward()
    optimizer.step()
    if step % 300 == 0:
        print(f"Step {step}, loss={loss.item():.4f}")


context = torch.tensor([[word2index["life"]]], dtype=torch.long)
out = model.generate(context, max_new_tokens=15)

print("\nGenerated text:\n")
print(" ".join(idx2word[int(i)] for i in out[0]))
