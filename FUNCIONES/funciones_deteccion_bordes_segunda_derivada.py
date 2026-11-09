import numpy as np

# FUNCIÓN GENERAR KERNEL LAPLACIANA DE LA GAUSSIANA
# ============================================================

def generar_kernel_log(sigma=None, W=None):
    """
    Genera un kernel LoG (Laplaciana de la Gaussiana)
    a partir de sigma o del tamaño W, usando la relación:
        W = 3 * c,  con  c = 2 * sqrt(2) * sigma  ⇒  W = 6 * sqrt(2) * sigma

    Si no se pasa ninguno de los dos parámetros,
    usa por defecto sigma = 2.0 y W = 17.

    Parámetros:
    ------------
    sigma : float, opcional
        Desviación típica de la Gaussiana.
    W : int, opcional
        Tamaño del kernel (impar). Si no se pasa, se calcula automáticamente.

    Devuelve:
    ----------
    LoG : np.ndarray
        Kernel Laplaciano de la Gaussiana (LoG) centrado en cero.
    sigma : float
        Desviación estándar usada.
    W : int
        Tamaño final del kernel (impar).
    """

    # --- Normalización de parámetros ---
    if sigma is None and W is None:
        # Valores por defecto
        sigma = 2.0
        W = 17

    elif sigma is None and W is not None:
        # Asegurar W impar antes de calcular sigma
        if W % 2 == 0:
            W += 1
        sigma = W / (6 * np.sqrt(2))

    elif W is None and sigma is not None:
        W = int(np.ceil(6 * np.sqrt(2) * sigma))
        if W % 2 == 0:
            W += 1

    # --- Generación del kernel ---
    r = W // 2
    y, x = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    norm = (x**2 + y**2) / (2 * sigma**2)
    LoG = (-1 / (np.pi * sigma**4)) * (1 - norm) * np.exp(-norm)

    # Normalizar para tener media cero
    LoG -= LoG.mean()

    return LoG


# FUNCIÓN DE DETECCIÓN DE CRUCES POR CERO
# ============================================================
def detectar_cruces_cero(L, umbral=None):
    """
    Detecta los cruces por cero (zero-crossings) en una imagen filtrada 
    (por ejemplo, salida de Laplaciana o Laplaciana de la Gaussiana).

    Parámetros:
    ------------
    L : np.ndarray
        matriz de entrada (respuesta Laplaciana o LoG, tipo float).
    umbral : float, opcional
        Magnitud mínima para considerar un cruce como borde. 
        Si es None, se calcula automáticamente: factor * media de |L|.
        Por defecto el factor es 0.75

    Devuelve:
    ----------
    zero_cross : np.ndarray (uint8)
        Imagen binaria (0 o 255) con los bordes detectados.
    umbral_usado : float
        Valor del umbral utilizado (automático o manual).
    """

    L = L.astype(np.float32)
    filas, cols = L.shape
    zero_cross = np.zeros_like(L, dtype=np.uint8)
    abs_L = np.abs(L)

    # --- Calcular umbral si no se proporciona ---
    factor_defecto = 0.75
    if umbral is None:
        umbral = factor_defecto * np.mean(abs_L)

    # --- Detectar cruces por cero ---
    for i in range(1, filas - 1):
        for j in range(1, cols - 1):
            patch = L[i - 1:i + 2, j - 1:j + 2]
            if patch.max() > 0 and patch.min() < 0:
                if abs_L[i, j] > umbral:
                    zero_cross[i, j] = 255

    return zero_cross, umbral
