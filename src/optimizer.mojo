trait Optimizer():
    fn step(self):
        ...
    fn zeroGrads(self):
        ...

struct SGD(Optimizer):
    pass

struct ADAM(Optimizer):
    pass
