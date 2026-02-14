from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
img = Image.open("신짱구.png").convert("L")

# NumPy 배열로 변환
img_np = np.array(img)

print(img_np.shape)   # (H, W, C)
H, W = img_np.shape
print(img_np.dtype)   # uint8


# plt.imshow(img_np)
# plt.axis("off")
# plt.show()

filter = np.array([[-1, -1, -1],
                   [0, 0, 0],
                   [1, 1, 1]])

def t(sub_img, filter):
    """
    sub_img : 3x3
    filter : 3x3
    """
    return np.dot(sub_img.flatten(), filter.flatten())


def conv(img, filter):
    H, W = img.shape     # 320, 259
    out_img = np.zeros((H - 2, W - 2))

    for i in range(H - 3 + 1):
        for j in range(W -3 + 1):
            out_img[i, j] = t(img[i:i+3, j:j+3], filter)
    return out_img

res = conv(img_np, filter)
print(res.shape)

plt.imshow(res)
plt.axis("off")
plt.show()