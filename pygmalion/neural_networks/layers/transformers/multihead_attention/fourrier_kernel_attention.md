kernelized attention

$$
A_{id} =  \sum_k \sum_j \left( \overline{Q}_{ik} \times \overline{K}_{jk} \times V_{jd} \right)
$$

kernelized attention with Fourier positional encoding

$$
A_{id} =  \sum_k \sum_j \left( \overline{Q}_{ik} \times \overline{K}_{jk} \times \Delta P_{ijk} \times V_{jd} \right)
$$

With 

$$
\Delta P_{ijk} = c_k \times cos \left(b_k + \sum_n a_{kn} \times (p_{in} - p_{jn}) \right)
$$

Which for a scalar $\vec{p} = t$, gives to $\Delta P_{ijk}$ the form of one harmonic of a Fourier series: $c_k \times cos \left(a_k \times \Delta t_{ij} + b_k \right)$. Hence the expressive power of this relative-positional attention mechanisme: The attention score calculated as $\sum_k \left( \overline{Q}_{ik} \times \overline{K}_{jk} \times \Delta P_{ijk} \right)$, as a set of functions containing the Fourier Series, can approximate any function of $\delta t_{ij}$ on a fixed time interval, for a given sequence of queries and keys, given a big enough embedding dimension.

To simplify the expression of $\Delta P_{ijk}$, we introduce $\hat{p}_{ik}$ and $\hat{p}_{jk}$ the linear projections from position dimension to embedding dimension.

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
sin(\hat{p}_i - \hat{p}_j) = cos(\hat{p}_j) \times sin(\hat{p}_i) - cos(\hat{p}_i) \times sin(\hat{p}_j) \\
cos(\hat{p}_i - \hat{p}_j) = cos(\hat{p}_i)\times cos(\hat{p}_j) + sin(\hat{p}_i) \times sin(\hat{p}_j)\\
\end{array}
\right.
$$

Which can be formulated

$$
\Delta P_{ijk} = c_k \times cos(\hat{p}_i) \times cos(\hat{p}_j) + c_k \times sin(\hat{p}_i) \times sin(\hat{p}_j)
$$

This allows us to compute the result of the attention operation without calculating the costly $\Delta P_{ijk}$ tensor:



## Linear complexity implementation