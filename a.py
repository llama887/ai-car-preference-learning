var = 1


def foo():
    global var
    var = 2


def balls():
    print(var)


if __name__ == "__main__":
    foo()
    balls()
