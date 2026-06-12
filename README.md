# How Large Language Models Work — Under the Hood

> A deep technical guide to understanding what actually happens inside GPT, Claude, and other LLMs — from raw text to transformer architecture, attention mechanisms, and how a model "thinks."

---

## Table of Contents

1. [What Is a Large Language Model?](#what-is-a-large-language-model)
2. [The Core Task: Next Token Prediction](#the-core-task-next-token-prediction)
3. [Tokenization — Breaking Text into Pieces](#tokenization--breaking-text-into-pieces)
4. [Embeddings — Turning Tokens into Numbers](#embeddings--turning-tokens-into-numbers)
5. [The Transformer Architecture](#the-transformer-architecture)
6. [Attention — The Key Mechanism](#attention--the-key-mechanism)
7. [How Attention Actually Works — Step by Step](#how-attention-actually-works--step-by-step)
8. [Multi-Head Attention](#multi-head-attention)
9. [Feed-Forward Layers](#feed-forward-layers)
10. [Training — How the Model Learns](#training--how-the-model-learns)
11. [RLHF — Making Models Helpful and Safe](#rlhf--making-models-helpful-and-safe)
12. [How Generation Works at Inference Time](#how-generation-works-at-inference-time)
13. [Context Windows and Memory](#context-windows-and-memory)
14. [GPT vs Claude vs Gemini — Key Differences](#gpt-vs-claude-vs-gemini--key-differences)
15. [Common Misconceptions](#common-misconceptions)
16. [Conclusion](#conclusion)

---

## What Is a Large Language Model?

A **Large Language Model (LLM)** is a neural network trained on massive amounts of text to predict and generate language. The "large" refers to the number of parameters — the adjustable numerical values that define the model's behavior.

| Model | Parameters | Training Data |
|---|---|---|
| GPT-2 (2019) | 1.5 billion | 40 GB of text |
| GPT-3 (2020) | 175 billion | 570 GB of text |
| GPT-4 (2023) | ~1 trillion (estimated) | Unknown |
| Claude 3 Opus | Unknown | Unknown |
| Llama 3 70B | 70 billion | 15 trillion tokens |

Despite their power, LLMs are fundamentally doing one thing:

> **Given all the text so far, what word (token) is most likely to come next?**

Everything — answering questions, writing code, reasoning through problems — emerges from this single capability applied at scale.

---

## The Core Task: Next Token Prediction

During training, the model sees text like:

```
"The capital of France is ___"
```

And learns to predict: `Paris`

Then:
```
"The capital of France is Paris. The Eiffel Tower is located in ___"
```

Predict: `Paris`

This is called **self-supervised learning** — the training data provides its own labels. No human needs to label anything; the model just predicts the next word, checks if it was right, and adjusts its parameters accordingly — billions of times.

The emergent result of doing this on enough text is a model that appears to understand language, reason, write code, translate, and much more.

---

## Tokenization — Breaking Text into Pieces

LLMs don't process characters or words — they process **tokens**. A token is a chunk of text, usually 3–4 characters on average for English.

### How Tokenization Works

Most modern LLMs use **Byte Pair Encoding (BPE)** — a compression algorithm that starts with individual characters and iteratively merges the most frequent pairs.

```
Original: "unhappiness"

Broken into tokens:
["un", "happ", "iness"]   → 3 tokens

Or possibly:
["un", "happiness"]        → 2 tokens
```

### Why Tokenization Matters

```python
import tiktoken  # OpenAI's tokenizer

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

text = "Hello, how are you?"
tokens = enc.encode(text)
print(tokens)        # [9906, 11, 1268, 527, 499, 30]
print(len(tokens))   # 6 tokens

# Interesting cases:
enc.encode("cat")        # [9864]          → 1 token
enc.encode("cats")       # [25413]         → 1 token
enc.encode("猫")         # [37861, 243]    → 2 tokens (Chinese needs more)
enc.encode("1+1=2")      # [16, 10, 16, 28, 17] → 5 tokens (math is expensive)
```

### Implications of Tokenization

- GPT-4 has a vocabulary of ~100,000 tokens
- Rare words, non-English text, and code often require more tokens
- The model never "sees" letters — it sees token IDs
- This is why LLMs sometimes fail at character-level tasks like "how many r's in strawberry?" — the model sees `["straw", "berry"]`, not individual characters

---

## Embeddings — Turning Tokens into Numbers

Neural networks can only process numbers. So each token ID is converted into an **embedding** — a high-dimensional vector of floating-point numbers.

```
Token: "Paris"   →   Token ID: 12366

Token ID: 12366  →   Embedding: [0.23, -0.87, 0.41, 1.12, ..., -0.33]
                                  ↑ 768, 1024, or 4096 numbers depending on model size
```

These numbers are not hand-crafted — they are learned during training. After training, semantically similar tokens end up with similar embeddings:

```
embedding("king")   - embedding("man") + embedding("woman") ≈ embedding("queen")
```

This is the famous "word2vec analogy" — meaning is encoded geometrically in embedding space. Words with similar meanings cluster together; words with opposite meanings point in opposite directions.

---

## The Transformer Architecture

The **Transformer** is the neural network architecture that powers all modern LLMs. Introduced in the 2017 paper "Attention Is All You Need" (Google Brain), it replaced recurrent neural networks (RNNs) and became the foundation of GPT, BERT, Claude, Gemini, and virtually every major LLM.

### High-Level Architecture

```
Input Text
    ↓
Tokenization
    ↓
Token Embeddings + Positional Encodings
    ↓
┌─────────────────────┐
│   Transformer Block │ ← repeated N times (e.g. 96 layers in GPT-4)
│                     │
│  ┌───────────────┐  │
│  │ Self-Attention│  │  ← "What other tokens should I pay attention to?"
│  └───────────────┘  │
│          ↓          │
│  ┌───────────────┐  │
│  │  Feed-Forward │  │  ← "Process and transform the attended information"
│  └───────────────┘  │
└─────────────────────┘
    ↓
Linear Layer + Softmax
    ↓
Probability distribution over all tokens
    ↓
Sample next token
```

Each transformer block refines the representation of every token, incorporating context from all other tokens via attention. After N layers, the final representation is projected into a probability distribution over the vocabulary — the model's prediction of what comes next.

---

## Attention — The Key Mechanism

**Attention** is the mechanism that allows every token to "look at" every other token in the sequence and decide how much to weight each one when building its representation.

Before transformers, RNNs processed text sequentially — token by token — and struggled to remember long-range dependencies. Attention solves this by letting every token directly attend to every other token, regardless of distance.

### The Intuition

Consider the sentence:

```
"The animal didn't cross the street because it was too tired."
```

What does "it" refer to? The animal, not the street. Humans understand this instantly. Attention gives the model a mechanism to figure this out — "it" should attend strongly to "animal" and weakly to "street."

### The Three Vectors: Q, K, V

For each token, the model creates three vectors:

- **Query (Q)** — "What am I looking for?"
- **Key (K)** — "What do I contain?"
- **Value (V)** — "What information do I carry?"

Attention between two tokens is computed by comparing one token's Query against another token's Key.

---

## How Attention Actually Works — Step by Step

Let's trace the math for a simple example.

**Input sentence:** `"cat sat on"`
**We want to compute attention for the token "sat"**

### Step 1: Create Q, K, V vectors

Each token's embedding is multiplied by three learned weight matrices (W_Q, W_K, W_V):

```
For "sat":
  Q_sat = embedding("sat") × W_Q  =  [1.0, 0.5]
  K_sat = embedding("sat") × W_K  =  [0.8, 0.3]
  V_sat = embedding("sat") × W_V  =  [0.9, 0.7]

For "cat":
  Q_cat = embedding("cat") × W_Q  =  [0.2, 0.8]
  K_cat = embedding("cat") × W_K  =  [0.6, 0.4]
  V_cat = embedding("cat") × W_V  =  [0.5, 0.2]
```

### Step 2: Compute Attention Scores

For each pair of tokens, compute the dot product of Q (query token) with K (key token):

```
Score("sat" → "cat") = Q_sat · K_cat = (1.0 × 0.6) + (0.5 × 0.4) = 0.8
Score("sat" → "sat") = Q_sat · K_sat = (1.0 × 0.8) + (0.5 × 0.3) = 0.95
Score("sat" → "on")  = Q_sat · K_on  = ...
```

### Step 3: Scale and Softmax

Scores are divided by √d_k (square root of key dimension) to prevent gradients from vanishing, then passed through softmax to create a probability distribution:

```
Raw scores:    [0.80, 0.95, 0.60]
After softmax: [0.31, 0.38, 0.24, ...]   ← sums to 1.0
```

These are the **attention weights** — how much "sat" should attend to each other token.

### Step 4: Weighted Sum of Values

The final output for "sat" is a weighted combination of all Value vectors:

```
Output("sat") = 0.31 × V_cat + 0.38 × V_sat + 0.24 × V_on + ...
```

This output vector now encodes "sat" enriched with context from all other tokens, weighted by relevance.

### The Full Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

This single formula is the heart of every LLM.

---

## Multi-Head Attention

Instead of computing attention once, transformers compute it **multiple times in parallel** with different weight matrices — each called an "attention head."

```
Input
  ↓
Split into H heads (e.g. 32 heads in GPT-3)

Head 1: Q1, K1, V1 → Attention_1  (might learn syntactic relationships)
Head 2: Q2, K2, V2 → Attention_2  (might learn semantic relationships)
Head 3: Q3, K3, V3 → Attention_3  (might learn coreference: "it" → "animal")
...
Head H: QH, KH, VH → Attention_H

Concatenate all heads → Linear projection → Output
```

Different heads specialize in different types of relationships. Some heads track subject-verb agreement; others track coreference; others track positional relationships. This specialization emerges naturally from training — it is not programmed.

---

## Feed-Forward Layers

After attention, each token passes through a **feed-forward network (FFN)** — two linear transformations with a non-linear activation in between:

```
FFN(x) = Linear_2(GELU(Linear_1(x)))
```

The FFN is applied **independently to each token** (unlike attention, which mixes information across tokens). Its role is to transform and process the attended information.

Interestingly, research suggests the FFN layers act as a kind of **key-value memory** — storing factual associations like "Eiffel Tower → Paris" that were learned during training.

The FFN is typically 4× wider than the attention layer. In GPT-3:
- Attention dimension: 12,288
- FFN dimension: 49,152

This makes FFN layers responsible for the majority of a model's parameters.

---

## Training — How the Model Learns

### Pre-Training

The model is trained on hundreds of billions to trillions of tokens of text from the internet, books, code repositories, and other sources.

**The objective:** minimize the average negative log-likelihood of predicting the correct next token:

```
Loss = -1/N × Σ log P(token_i | token_1, ..., token_{i-1})
```

In plain terms: **be surprised as little as possible by what comes next in real text.**

Training uses **stochastic gradient descent (SGD)** with **backpropagation** — the model makes a prediction, computes how wrong it was, and adjusts billions of parameters slightly in the direction that would have made it less wrong.

### Scale of Training

Training a frontier model is an enormous engineering and financial undertaking:

| Resource | Scale |
|---|---|
| Training tokens | 1–15 trillion |
| GPU hours | Millions |
| GPUs used | Thousands of H100s |
| Estimated cost | $50M–$100M+ |
| Training time | Months |

This is why only a handful of organizations can train frontier models from scratch.

### The Scaling Laws

OpenAI's 2020 "Chinchilla" paper established that model performance scales predictably with:

```
Performance ∝ (Parameters)^a × (Training Tokens)^b × (Compute)^c
```

A key finding: most models were undertrained. The optimal ratio is roughly **20 tokens of training data per parameter**. Llama 3's 70B model was trained on 15 trillion tokens — far more than the "optimal" to create a smaller but more capable model.

---

## RLHF — Making Models Helpful and Safe

A pre-trained LLM is a "document completer" — it predicts what text would plausibly follow its input. It doesn't naturally follow instructions, refuse harmful requests, or maintain a helpful persona.

**RLHF (Reinforcement Learning from Human Feedback)** transforms a raw language model into a useful assistant.

### The Three Stages

```
Stage 1: Supervised Fine-Tuning (SFT)
  ↓
Show the model examples of ideal conversations
(human-written prompt + ideal response pairs)
Fine-tune on these examples
Result: model that can follow instructions

Stage 2: Reward Model Training
  ↓
Show human raters multiple responses to the same prompt
Humans rank them: "Response A > Response C > Response B"
Train a separate "reward model" to predict human preferences

Stage 3: Reinforcement Learning (PPO)
  ↓
Generate responses to prompts
Score them with the reward model
Update the LLM to generate responses the reward model scores higher
Repeat millions of times
Result: model that optimizes for human-preferred responses
```

### Constitutional AI (Anthropic's Approach)

Claude uses a variant called **Constitutional AI (CAI)**. Instead of relying entirely on human preference labels, Anthropic defines a set of principles (a "constitution") and trains the model to:

1. Generate a response
2. Critique it against the constitution
3. Revise it
4. Use the revised responses as training signal

This reduces dependence on human labelers for safety-related feedback and creates more consistent behavior.

---

## How Generation Works at Inference Time

When you send a message to Claude or ChatGPT, here is what happens:

```
Step 1: Your message is tokenized
        "What is the capital of France?" → [3923, 374, 279, 6864, 315, 9822, 30]

Step 2: Tokens are embedded and fed through all transformer layers
        Each layer refines the representation
        The final layer outputs a probability distribution over ~100,000 tokens

Step 3: Sample the next token
        e.g., "The" (probability: 0.42)

Step 4: Append "The" to the sequence and repeat from Step 2
        Now predict the token after "The"

Step 5: Continue until:
        - A special [END] token is sampled
        - Maximum token limit is reached
        - Stop sequence detected
```

This is called **autoregressive generation** — the model generates one token at a time, feeding each output back as input.

### Sampling Parameters

The output can be controlled with several parameters:

| Parameter | Effect |
|---|---|
| **Temperature** | Higher = more random/creative; lower = more deterministic |
| **Top-p (nucleus sampling)** | Only sample from top tokens whose cumulative probability ≥ p |
| **Top-k** | Only sample from the top k most likely tokens |
| **Max tokens** | Hard limit on response length |

```
Temperature = 0.0  → Always pick the most likely token (deterministic)
Temperature = 1.0  → Sample proportionally to probability (default)
Temperature = 2.0  → Very random, often incoherent
```

---

## Context Windows and Memory

An LLM's **context window** is the maximum number of tokens it can "see" at once — its working memory.

| Model | Context Window |
|---|---|
| GPT-3 | 4,096 tokens (~3,000 words) |
| GPT-4 Turbo | 128,000 tokens (~96,000 words) |
| Claude 3 | 200,000 tokens (~150,000 words) |
| Gemini 1.5 Pro | 1,000,000 tokens |

### What "Context" Means

Every token in the context window attends to every other token via attention. This means:

- The model can reference anything said earlier in the conversation
- Longer contexts cost more compute (attention scales as O(n²) with sequence length)
- Beyond the context window, the model has **no memory whatsoever**

### Why LLMs Have No Persistent Memory

Each conversation starts fresh. The model has no memory of previous conversations — only what is in the current context window. This is a fundamental architectural limitation, not a design choice.

Systems like "memory" features work by retrieving relevant past conversations and inserting them into the context window — the model itself doesn't "remember" anything.

---

## GPT vs Claude vs Gemini — Key Differences

All frontier LLMs share the transformer foundation, but differ in training approach, safety philosophy, and architectural details.

| Aspect | GPT-4 (OpenAI) | Claude (Anthropic) | Gemini (Google) |
|---|---|---|---|
| Training approach | RLHF | Constitutional AI (CAI) + RLHF | RLHF + instruction tuning |
| Safety philosophy | Policy-based refusals | Principle-based reasoning | Policy-based refusals |
| Context window | 128K tokens | 200K tokens | 1M tokens (1.5 Pro) |
| Multimodal | Yes (vision, code) | Yes (vision) | Yes (vision, audio, video) |
| Open weights? | No | No | Partially (Gemma) |
| Architecture details | Largely secret | Largely secret | Largely secret |

### What Makes Claude Different

Anthropic's focus on **interpretability research** — understanding what happens inside the model — is unusual in the industry. Their Constitutional AI approach trains the model to reason about principles rather than just pattern-match to human preferences.

---

## Common Misconceptions

### "LLMs understand language"
LLMs manipulate statistical patterns in token sequences. Whether this constitutes "understanding" is a philosophical question. They produce outputs indistinguishable from understanding in many contexts — but they also fail in ways that suggest the underlying mechanism is different from human comprehension.

### "LLMs are just autocomplete"
Technically true at the mechanistic level — but dismissively so. Human intelligence could similarly be described as "just neurons firing." The emergent capabilities from predicting text at scale are qualitatively different from simple autocomplete.

### "Bigger is always better"
Not necessarily. Llama 3 8B, trained on enough data, outperforms GPT-3 175B on many benchmarks. Data quality, training duration, and RLHF quality matter as much as raw parameter count.

### "LLMs retrieve facts from memory"
LLMs don't retrieve — they generate. Factual knowledge is encoded in the weights as statistical associations. This is why they can "hallucinate" — generate fluent, plausible-sounding text that is factually wrong.

### "The model knows what it doesn't know"
Models are not inherently calibrated about their own uncertainty. They generate the most likely next token regardless of whether the underlying knowledge is solid or shaky. Uncertainty expressions ("I think," "I'm not sure") are learned patterns, not genuine epistemic states.

---

## Conclusion

Large Language Models are, at their core, extraordinarily large functions trained to predict text. But from that simple objective, scaled to trillions of parameters and tokens, emerge capabilities that continue to surprise even their creators.

**Key concepts from this article:**

- LLMs predict the next token — everything else is an emergent capability from doing this at scale
- Tokenization breaks text into subword chunks; the model never sees raw characters
- Embeddings map tokens to high-dimensional vectors where meaning is encoded geometrically
- The Transformer architecture uses **self-attention** to let every token look at every other token
- Attention computes relevance via Query-Key dot products, then applies weighted Value aggregation
- Multi-head attention runs multiple attention computations in parallel, each specializing in different relationships
- Training minimizes prediction error on massive text corpora via gradient descent
- RLHF and Constitutional AI align pre-trained models to be helpful, harmless, and honest
- Generation is autoregressive — one token at a time, each fed back as input
- Context windows are the model's working memory; beyond them, nothing is remembered

Understanding LLMs at this level demystifies the "magic" — and reveals a technology that is both more mechanical and more remarkable than popular discourse suggests.

---

## Further Reading

- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) — the original Transformer paper
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Anthropic's CAI paper
- [Chinchilla Scaling Laws](https://arxiv.org/abs/2203.15556)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — the best visual explanation
- [Andrej Karpathy — Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — hands-on implementation

---

## License

Released under the [MIT License](LICENSE). Free to use, share, and adapt.

---

*Found this useful? Star the repo ⭐ and share it with someone curious about how AI actually works.*
