import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
import re
import random

# 你的原始模型架構（保持不變）
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout, device):
        super().__init__()
        self.block_size = block_size
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate_short_sentences(self, idx, max_new_tokens=50, temperature=0.8, top_k=40, 
                                stop_tokens=None, max_sentence_length=30):
        """專門為短句生成優化的函數"""
        if stop_tokens is None:
            stop_tokens = ['.', '!', '?', '\n']  # 句子結束符號
        
        generated_tokens = 0
        sentence_length = 0
        
        for _ in range(max_new_tokens):
            if generated_tokens >= max_new_tokens:
                break
                
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # 如果句子太長，增加結束符號的機率
            if sentence_length > max_sentence_length:
                # 找到結束符號的索引並增加其機率
                for token_char in stop_tokens:
                    if hasattr(self, 'char_to_idx') and token_char in self.char_to_idx:
                        token_idx = self.char_to_idx[token_char]
                        logits[0, token_idx] += 2.0  # 增加結束符號機率
            
            # Top-k 採樣
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            generated_tokens += 1
            sentence_length += 1
            
            # 如果生成了結束符號，重置句子長度計數
            if hasattr(self, 'idx_to_char'):
                next_char = self.idx_to_char.get(idx_next.item(), '')
                if next_char in stop_tokens:
                    sentence_length = 0
                    
        return idx

    def generate_with_punctuation_boost(self, idx, max_new_tokens=50, temperature=0.8, 
                                      top_k=40, punctuation_boost=1.5):
        """增加標點符號機率的生成函數"""
        punctuation_chars = ['.', '!', '?', ',', ';', ':', '\n']
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # 增加標點符號的機率
            if hasattr(self, 'char_to_idx'):
                for punct in punctuation_chars:
                    if punct in self.char_to_idx:
                        punct_idx = self.char_to_idx[punct]
                        logits[0, punct_idx] *= punctuation_boost
            
            # Top-k 採樣
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
                    
        return idx

# 改良的資料預處理 - 專注於短句
def prepare_short_dialog_data():
    """專門為短句訓練準備的資料預處理"""
    print("Loading daily_dialog dataset for short sentence training...")
    dataset = load_dataset("daily_dialog", trust_remote_code=True)
    
    all_dialogs = []
    short_sentences = []
    
    for split in ['train', 'validation']:
        for dialog in dataset[split]['dialog']:
            for utterance in dialog:
                # 清理文本
                clean_utterance = re.sub(r'\s+', ' ', utterance.strip())
                clean_utterance = re.sub(r'[^\w\s\.,!?\-\']', '', clean_utterance)
                
                # 只保留相對較短的句子（10-50字元）
                if 10 <= len(clean_utterance) <= 50 and clean_utterance:
                    # 確保句子有適當的結尾
                    if not clean_utterance.endswith(('.', '!', '?')):
                        clean_utterance += '.'
                    
                    short_sentences.append(clean_utterance)
    
    print(f"Collected {len(short_sentences)} short sentences")
    
    # 隨機打散並組合成對話
    random.shuffle(short_sentences)
    
    # 將短句組合成小段對話（2-4句）
    dialog_groups = []
    for i in range(0, len(short_sentences), 3):
        group = short_sentences[i:i+3]  # 每組3句
        if len(group) >= 2:
            dialog_text = " ".join(group) + "\n"
            dialog_groups.append(dialog_text)
    
    full_text = "".join(dialog_groups)
    print(f"Created {len(dialog_groups)} dialog groups")
    print(f"Total characters: {len(full_text)}")
    
    return full_text

def save_model_simple(model, stoi, itos, config, filepath="short_dialog_model.pt"):
    """儲存模型"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'stoi': stoi,
        'itos': itos,
        'config': config
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def main():
    # 專為短句優化的超參數
    batch_size = 24 * 5  # 32      # 較大的batch size
    block_size = 24      # 較短的context window - 關鍵改變！
    max_iters = 40000    # 4000
    eval_interval = 200
    learning_rate = 1e-3  # 稍高的學習率
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 150
    n_embd = 64          # 較小的embedding - 避免過度擬合長句
    n_head = 4           # 較少的head
    n_layer = 4          # 較少的層數
    dropout = 0.1        # 增加dropout
    
    print(f"Using device: {device}")
    print(f"Optimized for SHORT sentence generation")
    print(f"Block size: {block_size} (shorter context)")
    torch.manual_seed(1337)
    
    # 載入專為短句準備的資料
    text = prepare_short_dialog_data()
    
    # 建立字符映射
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # 編碼資料
    data = torch.tensor(encode(text), dtype=torch.long)
    print(f"Encoded data shape: {data.shape}")
    
    # 訓練/驗證分割
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    def get_batch(split):
        data_split = train_data if split == 'train' else val_data
        ix = torch.randint(len(data_split) - block_size, (batch_size,))
        x = torch.stack([data_split[i:i+block_size] for i in ix])
        y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y
    
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    
    # 建立模型
    config = {
        'vocab_size': vocab_size, 'n_embd': n_embd, 'block_size': block_size,
        'n_head': n_head, 'n_layer': n_layer, 'dropout': dropout, 'device': device
    }
    model = BigramLanguageModel(**config)
    
    # 添加字符映射到模型（用於生成時的標點符號控制）
    model.char_to_idx = stoi
    model.idx_to_char = itos
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 優化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 訓練循環
    print("\nStarting training for short sentences...")
    best_val_loss = float('inf')
    
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                save_model_simple(model, stoi, itos, config)
        
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    print("\nTraining completed!")
    
    # 測試短句生成
    print("\n" + "="*60)
    print("SHORT SENTENCE GENERATION TESTING")
    print("="*60)
    
    model.eval()
    
    # 測試不同的短句生成策略
    test_prompts = [
        "Hello",
        "How are",
        "Good morning",
        "What do",
        "I think",
        "Can you"
    ]
    
    print("\n1. 標準短句生成 (max_tokens=70):")
    print("-" * 50)
    for prompt in test_prompts:
        try:
            context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
            generated = model.generate_short_sentences(
                context, 
                max_new_tokens=70,  # 限制較短
                temperature=0.7,    # 較低溫度
                top_k=30,          # 較小的top_k
                max_sentence_length=25
            )
            output = decode(generated[0].tolist())
            print(f"'{prompt}' -> {output[len(prompt):]}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n2. 增強標點符號生成:")
    print("-" * 50)
    for prompt in test_prompts[:3]:
        try:
            context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
            generated = model.generate_with_punctuation_boost(
                context,
                max_new_tokens=25,
                temperature=0.6,
                top_k=25,
                punctuation_boost=2.0
            )
            output = decode(generated[0].tolist())
            print(f"'{prompt}' -> {output[len(prompt):]}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n3. 超短句生成 (max_tokens=30):")
    print("-" * 50)
    for i in range(5):
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = model.generate_short_sentences(
            context,
            max_new_tokens=30,  # 非常短
            temperature=0.8,
            top_k=20,
            max_sentence_length=12
        )
        output = decode(generated[0].tolist())
        print(f"Sample {i+1}: {output}")
    
    return model, encode, decode

if __name__ == "__main__":
    model, encode, decode = main()