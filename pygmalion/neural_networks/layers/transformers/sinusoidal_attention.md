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
\Delta P_{ijk} = c_k \times 
sin \left(b_k + \sum_n a_{kn} \times (p_{in} - p_{jn}) \right)
$$

Which for a scalar $p$ resolves to one Harmonic of a Fourier series $c_k \times 
sin \left(a_k \times \Delta t + b_k \right)$.


It can also be reformulated as the sinus of a difference of two scalars:

$$
\left\{
\begin{array}{ll}
\Delta P_{ijk} = c_k \times sin(\hat{p}_i - \hat{p}_j) \\
\hat{p}_i = b_k + \sum_n a_{kn} \times p_{in} \\
\hat{p}_j = \sum_n a_{kn} \times p_{jn}
\end{array}
\right.
$$

The trigonometric identity allows us to split

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
\Delta P_{ijk} = c_k \times cos(\hat{p}_j) \times sin(\hat{p}_i) - c_k \times cos(\hat{p}_i) \times sin(\hat{p}_j)
$$