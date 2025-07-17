
# STAR: Strategy-Aware Refinement Module in Multitask Learning for Emotional Support Conversations

  

This repository implements the **STAR** module, a strategy-aware refinement component for multitask learning in emotional support conversations.  

It is built upon the [Emotional Support Conversation](https://github.com/thu-coai/Emotional-Support-Conversation/tree/main/codes_zcj) codebase.

  
  

---

  

## 🔧 Setup

  

We recommend using `conda` to create the virtual environment.

  

```bash

conda env create -f env.yaml

conda activate star

```

---

## 🧠 Model Overview

  

The **STAR** module is a pluggable refinement mechanism designed to operate on top of any decoder architecture.  

While our experiments are based on **BlenderBot-Joint**, STAR can be integrated into any decoder by attaching it after the decoder's output.

  

STAR introduces two core components:  

**Strategy-Aware Representation Adjustment (SARA)** and **Strategy Refinement (SR)**.

  
  

### Overall Architecture

  

Given a dialogue context, the decoder produces hidden states \( h \in \mathbb{R}^d \), and a predicted strategy token \( s \in \mathbb{N} \) is appended to guide generation.  

These signals are processed as follows:

  

---

  

### 1. Strategy-Aware Representation Adjustment (SARA)

  

SARA computes a global contextual summary using shared attention pooling:

  

\[

z = \text{Pooling}(h)

\]

  

The pooled vector \( z \) is passed through a two-layer feedforward network (with ReLU and Sigmoid activations) to compute a **gating value** \( g \in (0, 1) \):

  

\[

g = \sigma(f(z))

\]

  

This gating mechanism determines how much strategic adjustment should be injected into the final representation.

  

---

  

### 2. Strategy Refinement (SR)

  

The same vector \( z \) is separately transformed via another two-layer feedforward network into a refined strategy representation \( \hat{h} \):

  

\[

\hat{h} = P(z)

\]

  

The final hidden state used for response generation is then computed via a **gated fusion** of the strategy-aware and original decoder outputs:

  

\[

h' = g \odot \hat{h} + (1 - g) \odot h

\]

  
  

---

  

## 🚀 How to Run

  

### 1. Prepare Data

  

Download and preprocess the required datasets.

  

```bash

bash RUN/prepare_star.sh

```

  

### 2. Train the STAR Model

  

Train the model with multitask supervision and STAR refinement:

  

```bash

bash RUN/train_star.sh

```

  

### 3. Evaluate the Model

  

Run inference and compute evaluation metrics:

  

```bash

bash RUN/infer_star.sh

```


## Citation

If the code or data is used in your research, please star this repo and cite our paper as follows:

```

```