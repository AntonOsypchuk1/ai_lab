def or_gate(x1: int, x2: int) -> int:
    return x1 | x2

def and_gate(x1: int, x2: int) -> int:
    return x1 & x2

def xor_gate(x1: int, x2: int) -> int:
    or_result = or_gate(x1, x2)
    and_result = and_gate(x1, x2)
    return and_gate(or_result, ~and_result & 1)

for x1 in [0, 1]:
    for x2 in [0, 1]:
        result = xor_gate(x1, x2)
        print(f"XOR({x1}, {x2}) = {result}")
