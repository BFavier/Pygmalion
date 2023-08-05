class SpecialToken:
    """
    Special tokens for the <START>, <END>, <PAD>, ... tokens
    """
    def __repr__(self):
        return f"<{self.name}>"

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        is_token = issubclass(type(other), type(self))
        return is_token and (self.name == other.name)

    def __init__(self, name: str):
        self.name = name
