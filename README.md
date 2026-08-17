# 🎬 Movie Dekho Apne Marzi Ki

> **Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation**

A personal implementation of **K-RagRec**, exploring how **Knowledge Graphs + RAG + LLMs** can be used to build better movie recommendations.

The idea is to retrieve relevant movie knowledge from a KG, encode it with a GNN, and provide the useful context to an LLM for recommendation.

## 🧠 How it works

```text
User History
     ↓
Knowledge Graph
     ↓
Retrieve Sub-graphs
     ↓
GNN + Re-ranking
     ↓
LLM
     ↓
Movie Recommendation 🎬
```

## 🛠️ Setup

Tested with:

* Python 3.9
* PyTorch 2.4.1
* CUDA 11.8
* Transformers 4.45.2
* NetworkX 2.8.7
* PEFT 0.12.0

```bash
conda create -n kragrec python=3.9
conda activate kragrec

pip install numpy==1.23.4
pip install torch==2.4.1
pip install transformers==4.45.2
pip install networkx==2.8.7
pip install peft==0.12.0
```

## 📦 Dataset

Currently using **MovieLens-1M**.

Processed dataset + Knowledge Graph:

[Download from Google Drive](https://drive.google.com/file/d/1MlEPkRj47WrdXECUiz5D6Ie1oMv4hKC9/view?usp=sharing)

MovieLens-20M and Amazon Book can also be used after preprocessing.

## 🚀 Run

### Training

```bash
python train.py \
  --model_name graph_llm \
  --llm_model_name 7b \
  --llm_frozen True \
  --dataset ml1m \
  --batch_size 5 \
  --gnn_model_name gt \
  --gnn_num_layers 4 \
  --adaptive_ratio 0.5 \
  --sub_graph_numbers 3 \
  --reranking_numbers 5
```

### Evaluation

```bash
python evaluate.py \
  --model_name graph_llm \
  --llm_model_name 7b \
  --llm_frozen True \
  --dataset ml1m \
  --batch_size 5 \
  --gnn_model_name gt \
  --gnn_num_layers 4 \
  --adaptive_ratio 0.5 \
  --sub_graph_numbers 3 \
  --reranking_numbers 5
```

Or simply:

```bash
bash run.sh
```

## 🔬 Experiments

Things I'm interested in exploring:

* Different KG retrieval strategies
* GNN architectures
* LLM sizes
* Frozen vs. fine-tuned LLMs
* MovieLens-20M / Amazon Book
* Better recommendation explanations

## 📚 Reference

Based on the ACL paper:

**Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation**

> This is a personal/research implementation for experimentation and learning.

---
