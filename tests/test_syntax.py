import dis


def add_numbers(a, b):
    result = a + b
    return result


# 使用 dis.dis() 函数查看字节码
dis.dis(add_numbers)
