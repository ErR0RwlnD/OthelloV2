from multiprocessing import Pool


def foo(i):
    for _ in range(1000):
        print(i, end=' ')
    print('')


if __name__ == "__main__":
    pool = Pool(4)
    for i in range(8):
        pool.apply_async(foo, (i,))
    pool.close()
    pool.join()
