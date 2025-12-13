class A:
    def __init__(self) -> None:
        self.value = 42

    def create(self):
        self.test = 12


a = A()
print(a.value)
