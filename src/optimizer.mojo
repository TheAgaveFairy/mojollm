trait Optimizer:
    def step(self):
        ...

    def zeroGrads(self):
        ...


struct SGD(Optimizer):
    def step(self):
        pass

    def zeroGrads(self):
        pass


struct ADAM(Optimizer):
    def step(self):
        pass

    def zeroGrads(self):
        pass
