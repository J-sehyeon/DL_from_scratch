import numpy as np
import matplotlib.pyplot as plt

def plot_2dfunc(f, x_range, y_range, log=True, min_point=True, line=None, line_name=None, dot=False):
    """
    f           : get two args
    x_range     : list
    y_range     : list
    log         : scaling
    min_point   : show global minimum point
    line        : (N, 2), dtype=float
    line_name   : str
    """
    # 1) 그릴 범위 설정
    x_min, x_max = x_range
    y_min, y_max = y_range
    resolution = 200  # 촘촘할수록 부드러움

    # 2) 격자 생성
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)

    # 3) 함수값 계산
    Z = np.log1p(f(X, Y)) if log else f(X, Y)

    # 4) 등고선 플롯
    plt.figure(figsize=(6, 5))
    cs = plt.contour(X, Y, Z, levels=20)        # 선 등고선
    plt.clabel(cs, inline=True, fontsize=8)     # 등고선 값 라벨(원하면 제거)

    # 5) 최솟값 위치
    if min_point:
        min_idx = np.unravel_index(np.argmin(Z), Z.shape)
        plt.scatter(X[min_idx], Y[min_idx], color='red', marker="x", s=200, zorder=17)

    # 6) 선 플롯
    line = np.array(line)
    plt.plot(line[:, 0], line[:, 1], color="blue", linewidth=2, label=line_name, zorder=20)
    if dot:
        n = len(line) // 100
        plt.scatter(line[::n, 0], line[::n, 1], color="lightblue", marker="o", s=100, zorder=15, alpha=0.8)

    # 7) 채운 등고선(원하면)
    plt.contourf(X, Y, Z, levels=20, alpha=0.6)
    plt.colorbar(label="f(x, y)")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Contour plot of f(x, y)")
    plt.tight_layout()
    plt.show()