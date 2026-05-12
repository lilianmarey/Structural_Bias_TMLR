This anonymized repository provides Python code to reproduce experiments from the paper "Structural Bias Beyond Homophily: A Controlled Study of Fairness in Link Prediction", submitted to TMLR.

# Abstract

Graph link prediction (LP) plays a critical role in socially impactful applications such as job recommendation and friendship formation, making fairness in this task essential. While many fairness-aware methods manipulate graph structures to mitigate prediction disparities, the topological biases inherent to social graphs remain poorly understood and are often reduced to homophily alone. In this work, we study the relationship between structural biases and fairness outcomes in LP. To this end, we formalize a taxonomy of topological bias measures and introduce a graph generation method producing a diverse corpus of synthetic graphs with controlled structural properties. Using this corpus, we show empirically that fairness outcomes are strongly determined by graph topology, and that current fairness-aware methods remain sensitive to structural biases beyond homophily. These findings highlight the need for structurally grounded evaluations in fair graph learning.

For computing graphs, bias, and N2V, SVD, GCN embeddings and recommendations :  
```
python main.py
```

For analyzing results and creating figures :  
```
python analyze.py
```
