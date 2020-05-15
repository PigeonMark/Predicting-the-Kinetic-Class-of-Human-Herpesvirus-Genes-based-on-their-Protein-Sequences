import random
from collections import Counter

if __name__ == "__main__":
    labels = random.choices([1, 2, 3], [0.2, 0.3, 0.5], k=10**4)
    print(Counter(labels))
