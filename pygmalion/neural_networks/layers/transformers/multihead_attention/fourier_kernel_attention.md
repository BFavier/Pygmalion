<!--
kernelized attention

$$
A_{id} =  \sum_k \sum_j \left( \overline{Q}_{ik} \times \overline{K}_{jk} \times V_{jd} \right)
$$
-->

kernelized attention with Fourier positional encoding

This novel attention mechanism is defined similarly to a kernelized attention mecanisme :

$$
A_{id} =  \frac{\sum_j \left( S_{ij} \times V_{jd} \right)}{\sum_j S_{ij}}
$$

With $S_{ij}$ a matrix of attention scores between query $i$ and key $j$.

$$
S_{ij} = \sum_k \left( \overline{Q}_{ik} \times \overline{K}_{jk} \times \Delta P_{ijk} \right)
$$

Here $\overline{Q}_{ik} = \phi(Q_{ik})$ and $\overline{K}_{jk} = \phi(K_{jk})$ are the query and keys after application of the kernel function. The novelty comes from the $\Delta P_{ijk}$ term which encodes the relative position between key $i$ and query $j$ directly in the attention mecanism. It is defined as :

$$
\Delta P_{ijk} = c_k \times cos \left(b_k + \sum_n a_{kn} \times (p_{in} - p_{jn}) \right)
$$

Where $a$, $b$ and $c$ are tensors of learnable parameters, learned independently for each attention head.

For a scalar $\vec{p} = t$, $\Delta P_{ijk}$ takes the form of one harmonic of a Fourier series: $c_k \times cos \left(a_k \times \Delta t_{ij} + b_k \right)$. Hence the expressive power of this relative-positional attention mechanisme: The attention score calculated as $\sum_k \left( \overline{Q}_{ik} \times \overline{K}_{jk} \times \Delta P_{ijk} \right)$, as a superset of functions containing the Fourier Series, can approximate any function of $\Delta t_{ij}$ on a fixed time interval, for a given sequence of queries and keys, and given a big enough embedding dimension.

To simplify the expression of $\Delta P_{ijk}$ we introduce $\hat{p}_{ik}$ and $\hat{p}_{jk}$ the linear projections of $p_{in}$ and $p_{jn}$ respectively. These are projections from position dimension to embedding dimension.

$$
\left\{
\begin{array}{ll}
\Delta P_{ijk} = c_k \times cos(\hat{p}_{ik} - \hat{p}_{jk}) \\
\hat{p}_{ik} = b_k + \sum_n a_{kn} \times p_{in} \\
\hat{p}_{jk} = \sum_n a_{kn} \times p_{jn}
\end{array}
\right.
$$

The trigonometric identity allows us to split the cosinus of a difference into a sum of cos and sin products:

$$
\left\{
\begin{array}{ll}
sin(\hat{p}_i - \hat{p}_j) = \cos(\hat{p}_j) \times \sin(\hat{p}_i) - \cos(\hat{p}_i) \times \sin(\hat{p}_j) \\
cos(\hat{p}_i - \hat{p}_j) = \cos(\hat{p}_i)\times \cos(\hat{p}_j) + \sin(\hat{p}_i) \times \sin(\hat{p}_j)\\
\end{array}
\right.
$$

Which can be formulated

$$
\Delta P_{ijk} = c_k \times \cos(\hat{p}_i) \times \cos(\hat{p}_j) + c_k \times \sin(\hat{p}_i) \times \sin(\hat{p}_j)
$$

This allows us to compute the result of the attention operation without calculating the memory costly $\Delta P_{ijk}$ tensor:


## Linear complexity implementation

### Unmasked attention

The numerator of the attention operation can be expanded to

$$
A_{id} \times \sum_{j} S_{ij} =  \sum_k \sum_j \left( \overline{Q}_{ik} \times \overline{K}_{jk} \times c_k \times \cos(\hat{p}_i) \times \cos(\hat{p}_j) \times V_{jd} \right) \\
+ \sum_k \sum_j \left( \overline{Q}_{ik} \times \overline{K}_{jk} \times c_k \times \sin(\hat{p}_i) \times \sin(\hat{p}_j) \times V_{jd} \right)
$$

By regrouping the terms, it becomes clear that the summation over $j$ can be performed independently from the terms depending on $i$.

$$
A_{id} \times \sum_{j} S_{ij} =  \sum_k \left( c_k \times \overline{Q}_{ik} \times \cos(\hat{p}_i) \times \sum_j  \left( \overline{K}_{jk} \times \cos(\hat{p}_j) \times V_{jd} \right) \right) \\
+ \sum_k \left( c_k \times \overline{Q}_{ik} \times \sin(\hat{p}_i) \times \sum_j  \left( \overline{K}_{jk} \times \sin(\hat{p}_j) \times V_{jd} \right) \right)
$$

This means that, when calculating the sum over $j$ as an intermediary result, there is never a loop on $i$ (queries sequence length) and $j$ (key sequences length) at the same time. Hence the linear complexity with sequence length. The denominator of the attention operation can be expanded and computed with linear complexity the same way as the numerator:

$$
\sum_{j} S_{ij} =  \sum_k \left( c_k \times \overline{Q}_{ik} \times \cos(\hat{p}_i) \times \sum_j  \left( \overline{K}_{jk} \times \cos(\hat{p}_j) \right) \right) \\
+ \sum_k \left( c_k \times \overline{Q}_{ik} \times \sin(\hat{p}_i) \times \sum_j  \left( \overline{K}_{jk} \times \sin(\hat{p}_j) \right) \right)
$$

### masked attention

The masked futur variant of the attention is the same calculation except that the sum over $j$ is only performed for $j \leq i$.

$$
A_{id} \times \sum_{j} S_{ij} =  \sum_k \left( c_k \times \overline{Q}_{ik} \times \cos(\hat{p}_i) \times \sum_{j \leq i}  \left( \overline{K}_{jk} \times \cos(\hat{p}_j) \times V_{jd} \right) \right) \\
+ \sum_k \left( c_k \times \overline{Q}_{ik} \times \sin(\hat{p}_i) \times \sum_{j \leq i}  \left( \overline{K}_{jk} \times \sin(\hat{p}_j) \times V_{jd} \right) \right)
$$

This calculation can still be performed with linear complexity with regards to sequence length. The sum over $j \leq i$ is in fact the ith element of the cumulated sum of over $j$. If implemented correctly calculating a cumulated sum needs only one loop along the summed dimension. Thus the cumulated sum calculation is of complexity $O(L_K \times D^2)$ (looping over $j$, $k$ and $d$). And the second step of the calculation is of complexity $O(L_Q \times D^2)$ (looping over $i$, $k$ and $d$).

Finally the denominator can be calculated in the same fashion:

$$
\sum_{j} S_{ij} =  \sum_k \left( c_k \times \overline{Q}_{ik} \times \cos(\hat{p}_i) \times \sum_{j \leq i}  \left( \overline{K}_{jk} \times \cos(\hat{p}_j)\right) \right) \\
+ \sum_k \left( c_k \times \overline{Q}_{ik} \times \sin(\hat{p}_i) \times \sum_{j \leq i}  \left( \overline{K}_{jk} \times \sin(\hat{p}_j)\right) \right)
$$

Masking attention to tokens in the sequence of keys (for example padding tokens) can be achieved by setting their key vectors to $\vec{0}$