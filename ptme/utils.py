from pathlib import Path

STATE_DIR = Path(__file__).resolve().parents[1] / "data" / "ptme" / "heisenberg"

def ppt3(p2, p3):
    return p2 * p2 - p3

def D3(p2, p3):
    return -1/2 + 3/2 * p2 - p3

def D4(p2, p3, p4):
    return p4 - 1/2 * (1 - p2) * (1 - p2) + 1/3 - 4/3 * p3

def D5(p2, p3, p4, p5):
    return -(1/24 - 5/12 * p2 + 5/8 * p2 ** 2 + 5/6 * p3 - 5/6 * p2 * p3 - 5/4 * p4 + p5)

def B2(p2, p3, p4, p5):
    det = (-p2*p2*p5 + 2*p2*p3*p4 - p3*p3*p3 + p3*p5 - p4*p4)
    return -det


# 0.27, 0.39, 0.475
