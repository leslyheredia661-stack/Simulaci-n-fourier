# Simulaci-n-fourier

# Actividad formativa 2
# Simulacion y analisis de senales con la Transformada de Fourier

import numpy as np
import matplotlib.pyplot as plt

# 1. Configuracion de la simulacion
fs = 1000                 # Frecuencia de muestreo en Hz
T = 1.0                   # Duracion de la simulacion en segundos
t = np.arange(0, T, 1/fs) # Vector de tiempo
N = len(t)

# 2. Generacion de las senales elementales

# Pulso rectangular: vale 1 entre 0.2 y 0.4 segundos
pulso = np.where((t >= 0.2) & (t < 0.4), 1.0, 0.0)

# Funcion escalon: pasa de 0 a 1 en 0.5 segundos
escalon = np.where(t >= 0.5, 1.0, 0.0)

# Senal senoidal de 10 Hz
frecuencia = 10
senoidal = np.sin(2 * np.pi * frecuencia * t)

# 3. Funcion para calcular la transformada de Fourier
def analizar_senal(senal, nombre):
    # Se calcula la FFT y se normaliza la magnitud
    espectro = np.fft.rfft(senal)
    frecuencias = np.fft.rfftfreq(N, d=1/fs)
    magnitud = np.abs(espectro) / N

    # Correccion de amplitud para las frecuencias positivas
    if N % 2 == 0:
        magnitud[1:-1] *= 2
    else:
        magnitud[1:] *= 2

    # Fase en radianes
    fase = np.angle(espectro)

    # Graficas en el dominio del tiempo y de la frecuencia
    fig, ax = plt.subplots(3, 1, figsize=(10, 9))

    ax[0].plot(t, senal)
    ax[0].set_title(f"{nombre}: dominio del tiempo")
    ax[0].set_xlabel("Tiempo (s)")
    ax[0].set_ylabel("Amplitud")
    ax[0].grid(True)

    ax[1].stem(frecuencias, magnitud)
    ax[1].set_title("Espectro de magnitud")
    ax[1].set_xlabel("Frecuencia (Hz)")
    ax[1].set_ylabel("Magnitud")
    ax[1].set_xlim(0, 60)
    ax[1].grid(True)

    ax[2].plot(frecuencias, fase)
    ax[2].set_title("Espectro de fase")
    ax[2].set_xlabel("Frecuencia (Hz)")
    ax[2].set_ylabel("Fase (rad)")
    ax[2].set_xlim(0, 60)
    ax[2].grid(True)

    plt.tight_layout()
    plt.show()

    return frecuencias, magnitud, fase

# 4. Analisis de cada senal
f_pulso, mag_pulso, fase_pulso = analizar_senal(
    pulso, "Pulso rectangular"
)

f_escalon, mag_escalon, fase_escalon = analizar_senal(
    escalon, "Funcion escalon"
)

f_seno, mag_seno, fase_seno = analizar_senal(
    senoidal, "Senal senoidal de 10 Hz"
)

# 5. Verificacion de la propiedad de linealidad
senal_a = np.sin(2 * np.pi * 10 * t)
senal_b = 0.5 * np.sin(2 * np.pi * 20 * t)

fft_suma = np.fft.rfft(senal_a + senal_b)
suma_fft = np.fft.rfft(senal_a) + np.fft.rfft(senal_b)

error_linealidad = np.max(np.abs(fft_suma - suma_fft))
print("Error de linealidad:", error_linealidad)

# 6. Verificacion del desplazamiento en el tiempo
desplazada = np.roll(senoidal, 50)

fft_original = np.fft.rfft(senoidal)
fft_desplazada = np.fft.rfft(desplazada)

error_magnitud = np.max(
    np.abs(np.abs(fft_original) - np.abs(fft_desplazada))
)
print("Diferencia de magnitud al desplazar:", error_magnitud)

# 7. Comparacion de frecuencias: 10 Hz y 20 Hz
seno_10 = np.sin(2 * np.pi * 10 * t)
seno_20 = np.sin(2 * np.pi * 20 * t)

plt.figure(figsize=(10, 5))
plt.plot(t, seno_10, label="Senoidal de 10 Hz")
plt.plot(t, seno_20, label="Senoidal de 20 Hz", alpha=0.7)
plt.xlim(0, 0.3)
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Comparacion de senales con diferentes frecuencias")
plt.legend()
plt.grid(True)
plt.show()

print("Simulacion finalizada correctamente.")
