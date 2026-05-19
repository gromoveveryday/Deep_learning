# Основной код отсюда https://www.geeksforgeeks.org/nlp/negaitve-sampling-using-word2vec/

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader

def preprocess_corpus(text):
    words = text.split()          
    vocab = sorted(set(words)) 
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return words, word_to_idx, idx_to_word

def generate_training_data(words, word_to_idx, context_size):
    data = []
    for i in range(context_size, len(words) - context_size):
        target = word_to_idx[words[i]]
        for j in range(1, context_size + 1):
            context = word_to_idx[words[i - j]]
            data.append((target, context))
        for j in range(1, context_size + 1):
            context = word_to_idx[words[i + j]]
            data.append((target, context))
    return data

class Word2VecDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def get_negative_samples_batch(targets, num_negative, vocab_size):
    batch_size = len(targets)
    neg_samples = []
    for t in targets:
        samples = []
        while len(samples) < num_negative:
            s = np.random.randint(0, vocab_size)
            if s != t:
                samples.append(s)
        neg_samples.append(samples)
    return torch.LongTensor(neg_samples)

class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.target_emb = nn.Embedding(vocab_size, embedding_dim)
        self.context_emb = nn.Embedding(vocab_size, embedding_dim)
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self, target, context, negative_samples):
        target_emb = self.target_emb(target)                     # [B, D]
        context_emb = self.context_emb(context)                  # [B, D]
        neg_emb = self.context_emb(negative_samples)             # [B, neg, D]

        pos_score = self.log_sigmoid((target_emb * context_emb).sum(dim=1))   # [B]
        neg_score = self.log_sigmoid(-torch.bmm(neg_emb, target_emb.unsqueeze(2)).squeeze(2)).sum(dim=1)  # [B]
        loss = - (pos_score + neg_score).mean()
        return loss

def train_word_2_vec_with_neg_sampling(embedding_dim, context_size, num_negative_samples, learning_rate, num_epochs, batch_size):
    with open("generated_corpus.txt", "r", encoding="utf-8") as f:
        corpus_text = f.read()
    
    words, word_to_idx, idx_to_word = preprocess_corpus(corpus_text)
    training_data = generate_training_data(words, word_to_idx, context_size)
    dataset = Word2VecDataset(training_data)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    vocab_size = len(word_to_idx)
    model = SkipGramNegSampling(vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for target, context in dataloader:
            target = target.long()
            context = context.long()
            negative = get_negative_samples_batch(target.tolist(), num_negative_samples, vocab_size)

            optimizer.zero_grad()
            loss = model(target, context, negative)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    target_weights = model.target_emb.weight.detach().numpy()   # shape [vocab_size, emb_dim]
    word_embeddings = {word: target_weights[idx] for word, idx in word_to_idx.items()}
    return word_embeddings

word_embeddings = train_word_2_vec_with_neg_sampling(embedding_dim=10,
                                                     context_size=3,
                                                    num_negative_samples=5,
                                                    learning_rate=0.001,
                                                    num_epochs=7,
                                                    batch_size=32)

print(word_embeddings)