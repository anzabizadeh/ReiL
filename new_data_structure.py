class rl_data():
    def __init__(self):
        self.Q = 0
        self.N = 0
        self.E = 0

A = {s: {a: rl_data() for a in 'hi'} for s in 'tes'}
for s in 'te':
    A[s]['h'].E += 1
    A[s]['h'].N += 1
    A[s]['h'].Q += 0.1


max(A['t'][a].Q for a in A['t'].keys())

for s, a in A.items():
    print(a.E, a.N, a.Q)

print()