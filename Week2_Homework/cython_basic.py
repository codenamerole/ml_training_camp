import time
import numpy as np
import pandas as pd
import tm


def main():
    y = np.random.randint(2, size=(5000, 1))
    x = np.random.randint(10, size=(5000, 1))
    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])
    start = time.time()
    result = tm.target_mean_v3(data, 'y', 'x')
    end = time.time()
    print(end - start)


if __name__ == '__main__':
    main()
