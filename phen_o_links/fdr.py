class FDR(object):
    def __init__(self, d, field='P', alpha=0.05):
        self.v = True
        self.l = float(len(d))
        self.i = 0
        self.field = field
        self.alpha = alpha
        self.previous = None

    def __call__(self, S):

        if (self.previous is not None and self.previous > S[self.field]):
            raise ValueError("P-values are not sorted ({0}<{1})".format(
                S[self.field], self.previous))
        else:
            self.previous = S[self.field]

        if self.v:
            self.i += 1
            
            self.v = S[self.field] < self.alpha * self.i / self.l
        S['FDR'] = self.v
        return S

