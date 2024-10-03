class LSystems:
    def __init__(self, axiom: str, rule: str, angel: float, nesting_level: int, length: int):
        self.axiom = axiom
        self.rule = rule
        self.angle = angel
        self.length = length
        self.drawString = None

        self.prepare_draw_string(nesting_level)

    def prepare_draw_string(self, nesting_level):
        self.drawString = self.axiom
        tmp = []

        # In each iteration replace all Fs with rule
        for _ in range(nesting_level):
            for c in self.drawString:
                if c == 'F':
                    tmp.append(self.rule)
                else:
                    tmp.append(c)

            self.drawString = "".join(tmp)
            tmp.clear()
