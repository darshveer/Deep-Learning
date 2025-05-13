# Machine Learning

## Gradient Descent

Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In machine learning, gradient descent is often used to update the parameters of a model in order to minimize the loss function, and thus we take the negative of the gradient.

The gradient descent formula is:

$$w_{i+1} = w_i - \alpha \frac{\partial L}{\partial w_i}$$

where $w_i$ is the current set of model parameters, $\alpha$ is the learning rate, and $\frac{\partial L}{\partial w_i}$ is the gradient of the loss function with respect to the model parameters.

## Backpropagation

Backpropagation is an algorithm for supervised learning of artificial neural networks using gradient descent. Given an artificial neural network and an error function to be optimized (such as mean squared error), the algorithm computes the gradient of the error with respect to the neural network's weights and biases, and uses it to update the weights and biases to minimize the loss.

The backpropagation algorithm works by computing the error gradient of the loss function with respect to each of the weights by a method called the chain rule, computing the gradient one layer at a time, starting with the last layer. This is done by moving backwards through the network, adjusting the weights at each layer to minimize the loss. The algorithm terminates when the first layer is reached, at which point the gradients of the weights are used to update the weights.

Suppose we have $L$ layers, each layer having some $n_L$ neurons. We follow the following defintions:
- $w_{jk}^{(L)} \rightarrow$ The weight of the edge from the $k^{th}$ neuron in the $(L - 1)^{th}$ layer to the $j^{th}$ neuron in the $L^{th}$ layer. 
- $a_k^{(L)} \rightarrow$ The activation of the $k^{th}$ neuron in the $L^{th}$ layer.
- $b_k^{(L)} \rightarrow$ The bias of the $k^{th}$ neuron in the $L^{th}$ layer.
- $z_j^{(L)} \rightarrow$ The linear function of weights and biases.
- $\sigma (.) \rightarrow$ Some softening non-linear function. 
- $C_0 \rightarrow$ The cost function of the given model. Here, we use the MSE (Mean squared Error). Basically acts like a sum over the layer $L$.

> **Why Non Linear Functions?** $\\$
If we only used linear transformations, the neural network would essentially be a linear model, and the composition of linear functions would result in another linear function. This means the network would only be able to learn linear relationships, which is not sufficient for most real-world problems. Examples: Sigmoid, ReLU (Rectified Linear Unit), tanh, Leaky ReLU, Swish.

### Formulae:

- $$C_0 = \sum_{j=0}^{n_L-1} \left(a_j^{(L)} - y_j\right)^2$$
- $$z_j^{(L)} = b_j^{(L)} + \sum_k w_{jk}^{(L)} \ a_k^{(L-1)}$$
- $$a_j^{(L)} = \sigma \left(z_j^{(L)} \right)
- 

We now calculate the sensitivity of these weights and biases using partial differentiation to figure out how small changes in them affect the cost function.

1. $$\frac{\partial C_0}{\partial w_{jk}^{(L)}} = -2\left(a_j^{(L)} - y_j\right) \ a_k^{(L-1)} \ \sigma' \left(z_j^{(L)} \right)$$
2. $$\frac{\partial C_0}{\partial a_k^{(L-1)}} = -2 \ \sum_j w_{jk}^{(L)} \ \left(a_j^{(L)} - y_j\right) \ \sigma' \left(z_j^{(L)} \right)$$
3. $$\frac{\partial C_0}{\partial b_k^{(L)}} = -2 \ \sum_j \left(a_j^{(L)} - y_j\right) \ \sigma' \left(z_j^{(L)} \right)$$
4. $$\nabla C_0 = -2 \ \left(y - a^{(L)}\right) \ \left(a^{(L-1)}\right)^T \ \sigma' \left(z^{(L)}\right)$$
where formula 4 is the gradient of the cost function.
### Chain Rule:

We use the following chain rules to calculate the gradient of the cost with respect to the weights and biases.

1. $$\frac{\partial C_0}{\partial w_{jk}^{(L)}} = \frac{\partial C_0}{\partial a_j^{(L)}} \ \frac{\partial a_j^{(L)}}{\partial z_j^{(L)}} \ \frac{\partial z_j^{(L)}}{\partial w_{jk}^{(L)}}$$
2. $$\frac{\partial C_0}{\partial a_k^{(L-1)}} = \sum_j \left( \frac{\partial C_0}{\partial a_j^{(L)}} \ \frac{\partial a_j^{(L)}}{\partial z_j^{(L)}} \ \frac{\partial z_j^{(L)}}{\partial a_k^{(L-1)}} \right)$$
3. $$\frac{\partial C_0}{\partial b_k^{(L)}} = \frac{\partial C_0}{\partial a_k^{(L)}} \ \frac{\partial a_k^{(L)}}{\partial z_k^{(L)}} \ \frac{\partial z_k^{(L)}}{\partial b_k^{(L)}}$$

The formula 2 is summing over the given layer $L$ as each activation of the $(L-1)^{th}$ influences the activation of each neuron in th $L^{th}$ layer, so we add up all the influence of the activations of the previous layer.

# GPT and Transformers

- GPT stands for Geneative Pre-trained Transformer.
- Transformers are used in Deep Learning to give outputs based on context along with the weights and biases.

## Steps to Generation
1. Breaking the prompt into small sections called tokens. The accumulation of all tokens in mutliple dimensions is called a tensor.
2. Associate each word with a vector that somehow encodes the meaning of that token in numbers. The closer the vectors are in space, the more contextually similar the words are.
3. Pass the vectors through an attention block which allows the vectors to talk to each other and pass information between themselves and update values based on each other's contexts.
4. The vectors are then sent through a multilayer perceptron which basically parallely processes each vector and asks it questions to gain more context.
5. Repeat this multiple times based on requirements and essentially expect the meaning to be baked into the last vector of the sequence.
6. Perform a certain operation on the last vector to create a probability distribution of the next expected token.

---
- All these are basically huge vector operations.
- The vectors for tuning and trained using expected outputs for given inputs by methods such as Linear Regression (input vs. output as a 2-D grah). These are done using tunable parameters called weights.
- But, these parameters can be way more than 2.
- These are generally packaged together in a matrix-data product.

### 1. Embedding:
- The model is defined with some pre-defined vocabulary which contains of all the words/tokens. GPT-3 has about 50,000 words.
- Thee embedding matrix, $W_E$ has a column for each of these words which represent this word in numbers. It starts out random, but is updated based on data. Thus:
$$ \text{Words} \xrightarrow{\text{embedding}} \text{Vectors}$$
-  The embedding matrix gives how related a word is to the other words. Models like GPT-3 have 12,288 dimensions for each vector.
- We can, however project these vectors into lower dimension matrices to make it easier to work with. 

> **Try out:**
> ```python
> >> import gensim.downloader
> >> model = gensim.downloader.load("glove-wiki-gigaword-50")
> >> model["king"] - model["queen"] 
> ```
> **Output:**
> ```cmd
> array([ 0.12596998, -1.1372299 ,  0.66962993,  0.081499  ,  0.24217   ,
>       -0.73527   ,  0.08725001, -0.36390004, -0.561182  ,  0.44783002,
>       -0.30347598,  0.50713   , -0.640059  , -0.66753995,  0.39352003,
>        0.25708997,  0.25581998, -0.386451  ,  0.01601005,  0.45079002,
>       -0.38064003, -0.52175   ,  0.01098001, -0.58356   ,  0.13042   ,
>       -0.41770005, -0.03350008, -0.88575   ,  0.21020997,  0.389576  ,
>        0.4353    , -0.43530002, -0.39569002,  0.36874   ,  0.03794   ,
>        0.00517   , -0.18120003, -0.186358  , -0.01485997,  1.1987001 ,
>      -0.05844   ,  0.91547996, -0.83533   ,  0.51813   ,  0.0487    ,
>        0.54850996, -0.35146004,  0.6733    , -0.6535353 ,  0.09241998],
>      dtype=float32)
> ```
> - This basically make a row vector of 50 dimensions for each word. The closer the word in context, the lesser the difference in each dimension will be.
> - We can construct a vector called the context vector by subtacting the embedding vector of two words. For example, to create the context of plural words,
> $$\vec{\text{plur}} := E{(\text{cats})} - E(\text{cat})$$
> - This context vector can then be multiplied with some other word's embedding vector to get a value that predicts how plural the model finds a given word.

- So, the first embedding matriz is just a bunch of numbers, with no context whatsoever, taken from a look-up table. The network helps in given context through attention.
- The number of contexts for each model changes. GPT-3, for example, has 2,048 contexts (columns). So, the dimensions and size of its embedding matrix will be:
$$ 12,288 \ \text{dimensions} \ \times 50,257 \ \text{words} = 617,558,016$$
- The context size determines how much data a transformer can incorporate while making predictions.

### 2. Unembedding
- This required performing some operation on the last vector of the sequence to get a probability distribution.
- This can be done using an unembedding matrix, $W_U$ which has the dimensions for each of the 50,000 words, and then multipliying it by this vector to get the distribution after softmaxing the output. So, we get the dimensions as:
$$ 50,257 \ \text{words} \ \times 12,288 \ \text{dimensions} = 617,558,016$$
- This is similar to the embedding matrix, but with the order swapped.
- Again, $W_U$ begins random, but updates using data. 
- All the other vectors in the last matrix can be used to predict what word comes next based on the words before it.
> **Softmax:** $\\$
It is a way to turn an arbitrary list of numbers into a probability distribtution, such that the largest values end up closer to 1, and the smaller values end up closer to 0, such that all values sum up to 1.
>
> The formula for softmaxing an n-length column-vector $\mathbf{z}$ is:
> $$ \text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}} $$
>
>for $i = 1, 2, \ldots, n$, where $\mathbf{z} = [z_1, z_2, \ldots, z_n]^T$ and $e$ is the base of the natural logarithm.
>
> The formula for softmaxing an n-length column-vector $\mathbf{z}$ with a temperature of $T$ is:
> $$ \text{softmax}(\mathbf{z}, T)_i = \frac{e^{z_i/T}}{\sum_{j=1}^{n} e^{z_j/T}} $$
>
>for $i = 1, 2, \ldots, n$, where $\mathbf{z} = [z_1, z_2, \ldots, z_n]^T$ and $e$ is the base of the natural logarithm. $T$ influences how much weight is  given to the smaller values. If $T$ is large, the distribution is more uniform with smaller values getting more weight, and if $T$ is large, then the distribution is largely focused on the bigger values.

### 3. Attention
- After we tokenize our prompt, we can then query the prompt with certain questions. This query is already represented using a vector, called the query vector, $\vec{\mathbf{Q}}$. This has a smaller dimension than the embedding vector, say 128.
$$ \vec{\mathbf{Q}} = \mathbf{W}_Q \cdot \vec{\mathbf{E}} $$
- The values of $\mathbf{W_Q}$ are parameters of the model that are learnt from data.
- Now we need to create another vector called the key vector, $\vec{\mathbf{K}}$ from the input vector. The key vector is basically answering the query, in the same 128-dimensional space, for each token. This is also a linear transformation and is given by:
$$ \vec{\mathbf{K}} = \mathbf{W}_K \cdot \vec{\mathbf{E}} $$
- The values of $\mathbf{W_K}$ are also parameters of the model that are learnt from data.
- If $\vec{\mathbf{Q}}$ is closely aligned to a key $\vec{\mathbf{K}}$, it basically means that the key is the best answer to that query, meaning the embeddings of the token(s) of the query *attend to* to the embedding of the token(s) of the key. These are called attention weghts. More the attention weights, the better the output.
- The attention weights are given by the dot product of the query vector and the key vector and are divided by the square root of the dimension of the key vector, $d_k$.
$$ \text{Attention weights} = \frac{\vec{\mathbf{Q}} \cdot \vec{\mathbf{K}}^T}{\sqrt{d_k}} $$
- We then convert these attention weights into a probability distribution by using softmaxing. This creates a matrix called the **attention pattern**.
- The attention pattern is a matrix where the $i^{th}$ row and $j^{th}$ column is the attention weight of the $i^{th}$ query and the $j^{th}$ key.
- The attention pattern can be represented as $\mathbf{A}$ and is a matrix of dimensions $n \times m$ where $n$ is the number of queries and $m$ is the number of keys, such that $\mathbf{Q}$ is a $n \times d_k$ matrix and $\mathbf{K}$ is a $m \times d_k$ matrix. Generally, $n = m$, so we have a square matrix.
- The attention pattern can be given by the equation:
$$ \mathbf{A} = \text{softmax} \left( \frac{\mathbf{Q} \cdot \mathbf{K}^T}{\sqrt{d_k}} \right) $$
- The problem we now want to resolve is preventing the later words from affecting the previous ones, otherwise the answer to what comes nex will be revealed (because the attention matrix basically acts as many preditcions in one.)
- To solve this, we want the lower triangular part of the attention matrix to be all zeros as they indicate the effect of later tokens on previous ones. We do this by seeting this value to $-\infin$ before softmaxing so that the softmax automatically reduces all of them to 0. This is called **masking**.
- Finally, we need to generate the **value vector**, $\vec{\mathbf{V}}$, which is the actual output of the model. This is a linear transformation of the input vector, $\vec{\mathbf{E}}$.
$$ \vec{\mathbf{V}} = \mathbf{W}_V \cdot \vec{\mathbf{E}} $$
- The value vector is also a $m \times d_v$ matrix, where $m$ is the number of tokens in the input and $d_v$ is the dimension of the output of the model.
- The final output of the model is the dot product of the attention pattern and the value vector.
$$ \text{Output} = \mathbf{A} \cdot \mathbf{V} $$
- The output of the model is a $n \times d_v$ matrix, where $n$ is the number of queries and $d_v$ is the dimension of the output of the model.
- The value vector basically says that if I added $\vec{\mathbf{V}}$ to the embedding of one token based on its embedding vector, I will get the embedding vector of the token that is the value to the key of query. So, it adds more context by:
  - saying that if I have a query of "What is the next token", and I have a value vector where the value is the change in the embedding vector for that token, then I can predict the embedding vector of the token that is the value to the key of query.
  - In other words, the value vector is adding more context to each token by saying that if I have a query of "What is the next token", then I can predict the change in the embedding vector for that token.
  - We can represent this change in the embedding vector for each token as $\delta E_i$ as the sum of the products of the values and query of the value matrix, i.e.:
  $$ \delta E_i = \sum_j \mathbf{A}_{i,j} \cdot \mathbf{V}_{j,:} $$
  - The new embedding vector for each token is then calculated as:
  $$ \mathbf{E}'_i = \mathbf{E}_i + \delta \mathbf{E}_i $$
  - This is adding the context to the embedding vector for each token by adding the change in the embedding vector to the original embedding vector. This entire process is describing one *head* of attention.
  - This is also known as the "self-attention" layer, as it allows the model to attend to different parts of the input when generating the output for each token.
- In essence, the value map, $\mathbf{W}_V$ will thus have the dimensions of:
$$ 12,288 \ \text{dimensions} \ \times 12,288 \ \text{dimensions} = 150,994,944$$
- The dimensions of the key and query matrices, $\mathbf{W}_K$ and $\mathbf{W}_Q$ will be each:
$$ 128 \ \text{key-query space dim.} \ \times 12,288 \ \text{embedding dim.} = 1,572,864$$
- We don't, however, need all the dimension in the value map. Instead, we try to make it a low rank transformation such that:
$$ \# \ \text{of values} = \# \ \text{of keys} + \# \ \text{of queries}$$
- Thus, we can have a low rank transformation of the value matrix as:
$$ \mathbf{V} = \mathbf{U} \mathbf{S} \mathbf{V}^T$$
where $\mathbf{U}$ has the dimensions of:
$$ 128 \ \text{key-query space dim.} \ \times 128 \ \text{key-query space dim.} = 16,384$$
and $\mathbf{V}$ has the dimensions of:
$$ 128 \ \text{key-query space dim.} \ \times 12,288 \ \text{embedding dim.} = 1,572,864$$
and $\mathbf{S}$ is a diagonal matrix with the dimensions of:
$$ 128 \ \text{key-query space dim.} \ \times 128 \ \text{key-query space dim.} = 16,384$$
- In a cross-attention head the key and query space act on different datasets.
- A single attention block inside a transformer consists of multi-headed attention, where each of these operations is run in parallel, each with its own key-query-value maps. For example, GPT-3 uses 96 attention heads inside each attention block.
- The multi-headed attention block is defined as:
$$ \mathbf{M} = \text{concat}(\mathbf{M}_1, ..., \mathbf{M}_h) \mathbf{W}^O $$
where $\mathbf{M}_i$ is the output of the $i$th attention block, $\mathbf{W}^O$ is the output weights matrix, and $h$ is the number of heads.
- Each attention block is defined as:
$$ \mathbf{M}_i = \text{softmax}\left(\frac{\mathbf{Q}_i \mathbf{K}_i^T}{\sqrt{d_k}}\right) \mathbf{V}_i $$
where $\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i$ are the query, key and value matrices for the $i$th attention block, respectively, and $d_k$ is the dimensionality of the key space.
- The query, key and value matrices for the $i$th attention block are defined as:
$$ \mathbf{Q}_i = \mathbf{X} \mathbf{W}_i^Q $$
$$ \mathbf{K}_i = \mathbf{X} \mathbf{W}_i^K $$
$$ \mathbf{V}_i = \mathbf{X} \mathbf{W}_i^V $$
where $\mathbf{X}$ is the input matrix, and $\mathbf{W}_i^Q, \mathbf{W}_i^K, \mathbf{W}_i^V$ are the query, key and value weights matrices for the $i$th attention block, respectively.
- There are 96 such layers of attention blocks, interlaced with 96 layers of perceptrons, which wew will see next.


- Use of attention heads?
- Difference between value matrix and the key matrix with relation to the query.