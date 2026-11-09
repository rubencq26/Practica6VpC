import numpy as np
import cv2

def aplicar_supresion_no_maxima(Gx, Gy, Mag):
    ang = np.arctan2(Gy, Gx) * 180 / np.pi
    ang[ang < 0] += 180

    MagSnM = np.zeros_like(Mag, dtype=np.float32)

    for i in range(1, Mag.shape[0]-1):
        for j in range(1, Mag.shape[1]-1):
            # Determinar vecinos en la dirección del gradiente
            if (0 <= ang[i, j] < 22.5) or (157.5 <= ang[i, j] <= 180):
                q = Mag[i, j+1]
                r = Mag[i, j-1]
            elif (22.5 <= ang[i, j] < 67.5):
                q = Mag[i+1, j-1]
                r = Mag[i-1, j+1]
            elif (67.5 <= ang[i, j] < 112.5):
                q = Mag[i+1, j]
                r = Mag[i-1, j]
            else:  # 135°
                q = Mag[i-1, j-1]
                r = Mag[i+1, j+1]

            if (Mag[i, j] >= q) and (Mag[i, j] >= r):
                MagSnM[i, j] = Mag[i, j]

    return MagSnM
