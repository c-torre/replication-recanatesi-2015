from deco import concurrent, synchronized
from time import sleep


@concurrent
def slow(index):
    sleep(5)
    return index


@synchronized
def run():
    a = []
    for index in list('123'):
        a.append(slow(index))

    return a


def main():

    print(run())


if __name__ == "__main__":
    main()
