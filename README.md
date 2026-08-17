# Movie dekho apne marzi ki
This is the pytorch implementation for our ACL paper "Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation".

Environment
Python==3.9
numpy==1.23.4
torch==2.4.1
cuda==11.8.89
transformers==4.45.2
networkx==2.8.7
peft==0.12.0
Dataset
We provide three datasets: MovieLens-1M, MovieLens-20M and Amazon Book.

Due to space limitations, we only processed MovieLens-1M and put the KG and datasets at https://drive.google.com/file/d/1MlEPkRj47WrdXECUiz5D6Ie1oMv4hKC9/view?usp=sharing. Other datasets we provided raw data that can be processed as in the paper.

KG
We process the knowledge graph and provide the corresponding knowledge vector databases.

An example to run K-RagRec on MovieLens-1M
To run K-RagRec on MovieLens-1M with threshold 
p
=
50
, retrieve knowledge sub-graphs numbers 
k
=
3
, and re-ranking knowledge sub-graphs numbers 
K
=
3
, respectively.

For Training:

python train.py --model_name graph_llm --llm_model_name 7b --llm_frozen True --dataset ml1m --batch_size 5 --gnn_model_name gt --gnn_num_layers 4 --adaptive_ratio 0.5 --sub_graph_numbers 3 --reranking_numbers 5--adaptive_ratio 5 
For evaluation:

python evaluate.py  --model_name graph_llm --llm_model_name 7b --llm_frozen True --dataset ml1m --batch_size 5 --gnn_model_name gt --gnn_num_layers 4 --adaptive_ratio 0.5 --sub_graph_numbers 3 --reranking_numbers 5--adaptive_ratio 5 
or you can run:

bash run.sh
