import numpy as np
import matplotlib.pyplot as plt

# 1. Configuración de datos
t = np.linspace(0, 10, 100)
sequence_x = 0.3 * np.sin(2 * np.pi * t / 4) + 2.0
sequence_y = 0.3 * np.sin(2 * np.pi * (t - 0.6) / 4)

# 2. Definición Manual de Puntos Clave
align_points = [
    ((1.0, 2.3), (1.6, 0.3)),   # Pico 1
    ((3.0, 1.7), (3.6, -0.3)),  # Valle 1
    ((5.0, 2.3), (5.6, 0.3)),   # Pico 2
    ((7.0, 1.7), (7.6, -0.3)),  # Valle 2
    ((9.0, 2.3), (9.6, 0.3))    # Pico 3
]

# 3. Visualización (Ajuste de proporciones y márgenes)
plt.figure(figsize=(12, 2.5), dpi=200)

# Dibujar las series
plt.plot(t, sequence_x, 'k-', linewidth=1.2)
plt.plot(t, sequence_y, 'k--', linewidth=1.2)

# Dibujar las líneas de alineación (Flechas Rojas)
for (start, end) in align_points:
    plt.annotate("", xy=end, xytext=start,
                 arrowprops=dict(arrowstyle="<->", color='red', lw=0.8, alpha=0.5))

# Etiquetas de las secuencias
plt.text(-0.8, 2.0, 'Serie X', fontfamily='serif', fontsize=10, ha='right', va='center')
plt.text(-0.8, 0.0, 'Serie Y', fontfamily='serif', fontsize=10, ha='right', va='center')

# --- CORRECCIÓN DEL EJE TIME ---
# Dibujamos la línea del eje X manualmente
plt.plot([0, 10.5], [-1.5, -1.5], 'k-', linewidth=1) 
# Añadimos la punta de la flecha al final de la línea
plt.annotate("", xy=(10.7, -1.5), xytext=(10.5, -1.5),
             arrowprops=dict(arrowstyle="->", color='black', lw=1))
# Ponemos el texto "Time" desplazado a la derecha para que no choque
plt.text(11.0, -1.5, 'Tiempo', fontfamily='serif', fontsize=10, va='center', ha='left')

# 4. Ajuste de límites (Más aire para evitar el efecto zoom)
plt.xlim(-4, 13)
plt.ylim(-2.5, 3.5)
plt.gca().axis('off')

plt.tight_layout()
plt.savefig('dtw_alignment_fixed_time.png', bbox_inches='tight', dpi=300)
plt.show()