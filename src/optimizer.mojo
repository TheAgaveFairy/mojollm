trait Optimizer:
    fn step(self):
        ...

    fn zeroGrads(self):
        ...


struct SGD(Optimizer):
    fn step(self):
        pass

    fn zeroGrads(self):
        pass


struct ADAM(Optimizer):
    fn step(self):
        pass

    fn zeroGrads(self):
        pass
