import numpy as np
from tabpfn import TabPFNClassifier

X = np.random.randn(50, 5)
y = np.random.randint(0, 2, size=50)

m = TabPFNClassifier(device="cpu")
m.fit(X, y)
print("Pesos OK / cacheados")
