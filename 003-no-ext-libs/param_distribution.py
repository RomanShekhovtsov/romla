import enum


class ParamDistributionStrategy(enum):
    uniform = 'uniform'
    local_min = 'local_min'


class ParamDistribution:

    name = None
    space = None
    strategy = None

    used_values = None

    def __init__(self, name, space, strategy=ParamDistributionStrategy.uniform):
        self.name = name
        self.space = space
        self.strategy = strategy

