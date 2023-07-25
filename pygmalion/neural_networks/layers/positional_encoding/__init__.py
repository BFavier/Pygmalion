from typing import Union as _Union
from typing import Type as _Type
from ._learned_positional_encoding import LearnedPositionalEncoding
from ._sinusoidal_positional_encoding import SinusoidalPositionalEncoding

POSITIONAL_ENCODING_TYPE = _Union[_Type[LearnedPositionalEncoding], _Type[SinusoidalPositionalEncoding]]
