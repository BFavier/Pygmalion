class MONOTONICITY(int):

    def __repr__(self):
        return "MONOTONICITY."+("INCREASING" if self == 1 else "DECREASING")

    def __new__(cls, value):
        assert value in {-1, 1}
        return super().__new__(cls, value)
    
    INCREASING: "MONOTONICITY" = None
    DECREASING: "MONOTONICITY" = None


MONOTONICITY.INCREASING = MONOTONICITY(1)
MONOTONICITY.DECREASING = MONOTONICITY(-1)