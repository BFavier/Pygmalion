from typing import Union as _Union
from typing import Type as _Type
from ._fourier_kernel_attention import FourrierKernelAttention
from ._kernelized_attention import KernelizedAttention
from ._scaled_dot_product import ScaledDotProductAttention

ATTENTION_TYPE = _Union[_Type[ScaledDotProductAttention], _Type[KernelizedAttention], _Type[FourrierKernelAttention]]