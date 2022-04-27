# Analysis of Graph Neural Networks on Amazon Co-Purchase Graph
Authors: Vincent Wilmet, Suer Lin, Lu Wang, Asrorbek Orzikulov

The multi-class classification problem has been here for decades, and many algorithms have been developed to deal with this task. Formerly, the data used for a classification task was in a tabular form. However, in the last 10 years, various attempts were done to develop and test classification models that work on graph data. In this study, we compared 5 such graph neural networks on a very challenging dataset, the Amazon product co-purchasing data. To make the task more realistic, we utilized a sales-based split, using more popular products as training examples and less frequent ones as a test set. Our experiments show that adaptive Graph Attention Networks perform the best and have the narrowest gap between the training accuracy and the test accuracy. Meanwhile, models using simple Graph Convolutional Layers had the lowest accuracy and the widest generalization gap

We built 5 different models using graph convolutional layers that are popular in the academy and the industry. Each model contained only one type of layer, and attention-based models also had skip
connections. Below, we presented a brief description for each of those layers.

\textbf{GCN} : Our baseline model includes the popular Graph Convolutional Layers. For each node, we need to consider all its neighbors and the characteristic information they contain. Assuming that we use the function Average(), we can do this for each node and get an average representation that can be input into the neural network.

Each GCN layer has the following information propagation rule:
H^{\left(l+1 \right)} = \sigma \left(\tilde D^{-\frac{1}{2}} \tilde A \tilde D^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)

where $\tilde A = A + I_N$ is the adjacency matrix with self-connections, $\sigma$ is a non-linear activation function, $\tilde D_{ii} = \sum_j \tilde A_{ij}$ and $W^{(l)}$ are trainable weight matrices. Since the equation is computationally expensive, it is estimated using a localized first-order approximation. This is the same equation as the traditional GCN convolutional step \mathbf{H} =  \mathbf{D}^{-\frac{1}{2}} \mathbf{A} \mathbf{D}^{-\frac{1}{2}} \mathbf{X} \mathbf{W}.

\textbf{GraphSAGE} : GraphSAGE (SAmple and aggreGatE), uses SAGEConv and unlike embedding approaches that are based on matrix factorization, it leverages node features (e.g., text attributes, node profile information, node degrees) in order to learn an embedding function that can generalize to unseen nodes. It is developed on the basis of revolution and has the capability of generalization for unseen data. In the process of training, the neighbor nodes are sampled instead of training all nodes. The GraphSage algorithm exploits both the rich node features and the topological structure of each node’s neighbourhood simultaneously to efficiently generate representations for new nodes without retraining.

\textbf{GAT}: The GATConv is a foundational pillar of Graph Attention Networks (GAT). it uses masked self-attention layers to solve the problem of convolution of the current image. The characteristics of neighbor nodes can be aggregated by stacking layers without the need for complex matrix operations nor knowing the entire graph structure upfront. 
Computationally, it is highly efficient: the operation of the self- attentional layer can be parallelized across all edges, and the computation of output features can be parallelized across all nodes. No eigendecompositions or similar costly matrix operations are required. The time complexity of a single GAT attention head computing F’ features may be expressed as O( |V |FF’ + |E|F’ ), where F is the number of input features, and |V | and |E| are the numbers of nodes and edges in the graph, respectively.


\textbf{LeGCN}: Local Extrema Convolutional layer (LeConv) is an extension of the simple GCN layer that finds the importance of nodes with respect to their neighbors using the difference operator. Internally, a GCN layer computes the importance 𝜙 = 𝑋𝑊 for each neighbour of a node and subsequently uses a weighted average over all neighbours. If a score of one node is very high, the scores of all its neighbours will be high too, because of the weighted averaging. On the other hand, LeConv uses the following rule to assign scores to neighbours of a node, which allows the network to select local extremas of neighbour scores and use a better aggregation over neighbourhood methods.

\phi_i = \sigma \left(x_i W_1 + \sum_{j \in \mathcal{N}_{(i)}} A_{ij} \left(x_i W_2 - x_j W_3\right) \right)

\textbf{GATv2}: The model is developed by Brody et. al, who saw there was room for improvement in GAT model. Since the linear layers (matrix product with the weight matrix and the dot product with attention coefficients) in the standard GATConv are applied right after each other, they can be collapsed into a single linear operation. As a result, the ranking of attended nodes is unconditioned on the query node. Said otherwise, the ranking (the argsort) of attention coefficients is shared (static) across all nodes in the graph. That’s why the expressiveness power of simple GATConv layers is restricted. GATv2Conv fixes this problem by introducing adaptive attention mechanism, that allows the rankings of attention coefficients to differ from one node to another. Due to more flexible attention ratings, the GATv2Conv-based models have a greater expressiveness power and hence perform better. 
The exact mechanism of this "fixing" is shown below. In GAT-Conv layers, the dot product of attention coefficients and hidden representations are computed before a non-linearity (usually a Leaky ReLU). In GATv2Conv, the dot product is computed after a non-linearity is applied to a hidden representation.
\begin{align} \text{GATConv:} \quad e(\mathbf h_i, \mathbf h_j) = \sigma\left( \mathbf{a}^T \left[\mathbf W \mathbf h_i || \mathbf W \mathbf h_j \right] \right)
\end{align}
\begin{align} 
\text{GATv2Conv:} \quad e(\mathbf h_i, \mathbf h_j) = \mathbf{a}^T\sigma\left( \mathbf W\left[\mathbf h_i || \mathbf h_j \right] \right)
\end{align}



