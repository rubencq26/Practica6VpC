import numpy as np
import cv2

def calcular_gradiente(I, Hx, Hy):
    Hx = np.asarray(Hx, dtype=np.float64)
    Hy = np.asarray(Hy, dtype=np.float64)

    Gx = cv2.filter2D(I, cv2.CV_64F, Hx, borderType=cv2.BORDER_REPLICATE)
    Gy = cv2.filter2D(I, cv2.CV_64F, Hy, borderType=cv2.BORDER_REPLICATE)

    ModG = np.sqrt(Gx**2 + Gy**2)      # o cv2.magnitude(Gx, Gy)

    return Gx, Gy, ModG