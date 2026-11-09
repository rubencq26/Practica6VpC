import numpy as np

def reescalar_valores_matriz_a_imagen(matriz, pMin=None, pMax=None):
    """
    Reescala matriz a imagen entre 0-255 tipo uint8 de forma lineal

    Parámetros:
    -----------
    matriz : np.ndarray
        matriz de entrada (tipo float o entero).
    pMin, pMmax : float (opcionales)
        Valores mínimo y máximo del rango de escala de entrada.
        Si no se especifican, se usan el mínimo y máximo de la matriz.

    Devuelve:
    ----------
    img : np.ndarray
        Imagen uint8 con valores entre 0 y 255.
    """
    matriz = matriz.astype(float)

    if pMin is None:
        pMin = np.min(matriz)
    if pMax is None:
        pMax = np.max(matriz)

    # Evita división por cero si la matriz tiene valores constantes
    if pMax == pMin:
        return np.zeros_like(matriz, dtype=np.uint8)
    else:
        qMin, qMax = 0, 255
        img = qMin + ((qMax - qMin) / (pMax - pMin)) * (matriz - pMin)
        return np.clip(np.round(img), 0, 255).astype(np.uint8)