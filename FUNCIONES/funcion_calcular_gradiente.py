import numpy as np
import cv2

def calcular_gradiente(I, Hx, Hy):
    filas, columnas = I.shape
    nx, mx = Hx.shape
    ny, my = Hy.shape

    mitadx = [nx // 2, mx // 2]
    mitady = [ny // 2, my // 2]

    padx = cv2.copyMakeBorder(I, mitadx[0], mitadx[0], mitadx[1], mitadx[1], borderType=cv2.BORDER_REPLICATE)
    pady = cv2.copyMakeBorder(I, mitady[0], mitady[0], mitady[1], mitady[1], borderType=cv2.BORDER_REPLICATE)

    Gx = np.zeros_like(I, dtype=np.float32)
    Gy = np.zeros_like(I, dtype=np.float32)
    ModG = np.zeros_like(I, dtype=np.float32)

    for i in range(filas):
        for j in range(columnas):
            ventanax = padx[i:i+nx, j:j+mx]
            ventanay = pady[i:i+ny, j:j+my]
            Gx[i,j] = np.sum(ventanax.astype(np.float32) * Hx)
            Gy[i,j] = np.sum(ventanay.astype(np.float32) * Hy)

    ModG = np.sqrt(Gx**2 + Gy**2)

    return Gx, Gy, ModG