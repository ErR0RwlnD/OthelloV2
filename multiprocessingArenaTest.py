from multiprocessing import Pool


class testA():
    def __init__(self, n):
        self.n = n

    def plus(self):
        self.n += 1

    def get(self):
        return self.n, self.n+1, self.n+2


def foo(n):
    a = testA(n)
    a.plus()
    return a.get()


if __name__ == "__main__":
    pool = Pool(8)
    b = pool.imap(foo, [1]*8)
    pool.close()
    pool.join()
    print(sum(*b))
