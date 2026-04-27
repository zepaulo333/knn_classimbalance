import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon
from scipy.spatial import distance_matrix
import itertools

# 1. Preparar os dados para ilustrar os 4 casos estruturais do texto
np.random.seed(42)
# Caso 1: Clean Majority (nuvem densa da maioria)
clean_maj = np.random.randn(15, 2) * 0.25 + [0, 2.5]
# Caso 2: Clean Minority (nuvem densa da minoria)
clean_min = np.random.randn(8, 2) * 0.15 + [3.5, 2.5]
# Caso 3: Boundary Region (maioria e minoria intercaladas)
boundary_maj = np.random.randn(12, 2) * 0.35 + [1.5, 0]
boundary_min = np.random.randn(12, 2) * 0.35 + [2.0, 0]
# Caso 4: Isolated Outlier (um ponto da minoria rodeado pela maioria, ou longe)
outlier = np.array([[-1.5, 0]])

# Juntar tudo
majority = np.vstack([clean_maj, boundary_maj])
minority = np.vstack([clean_min, boundary_min, outlier])
points = np.vstack([majority, minority])

dist_mat = distance_matrix(points, points)

# 2. Configurar a figura
fig, ax = plt.subplots(figsize=(10, 8))

def update(frame):
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim(-2.5, 4.5)
    ax.set_ylim(-1.0, 3.5)
    
    r = frame
    ax.set_title(f"Filtração de Vietoris-Rips | Crescimento do Raio (ε) = {r:.2f}", fontsize=14, pad=20)
    
    # Adicionar legendas estáticas para as 4 regiões
    ax.text(0, 3.2, 'Clean Majority', ha='center', color='blue', fontweight='bold')
    ax.text(3.5, 3.0, 'Clean Minority', ha='center', color='red', fontweight='bold')
    ax.text(1.75, -1.0, 'Boundary Region\n(Loops / H1 Events)', ha='center', fontweight='bold')
    ax.text(-1.5, -0.3, 'Isolated\nOutlier', ha='center', color='red', fontweight='bold')

    # Desenhar os limites de influência (bolhas)
    for p in points:
        ax.add_patch(Circle(p, radius=r, color='gray', alpha=0.1, zorder=1))
        
    # Desenhar conexões (Arestas / 1-simplexos)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if dist_mat[i, j] <= 2 * r:
                ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 
                        color='black', linewidth=1.5, alpha=0.5, zorder=2)
                
    # Desenhar Triângulos preenchidos (Faces / 2-simplexos)
    # Se 3 pontos estão todos conectados entre si, preenche-se o triângulo
    for i, j, k in itertools.combinations(range(len(points)), 3):
        if dist_mat[i,j] <= 2*r and dist_mat[j,k] <= 2*r and dist_mat[i,k] <= 2*r:
            tri = Polygon([points[i], points[j], points[k]], 
                          closed=True, color='mediumpurple', alpha=0.4, zorder=3)
            ax.add_patch(tri)

    # Desenhar os pontos por cima de tudo
    ax.scatter(majority[:, 0], majority[:, 1], c='dodgerblue', s=60, edgecolors='black', label='Maioria', zorder=5)
    ax.scatter(minority[:, 0], minority[:, 1], c='crimson', s=60, edgecolors='black', label='Minoria', zorder=5)
    ax.legend(loc='upper left')

# 3. Criar a animação (80 frames, do raio 0 até 0.6)
frames_radius = np.linspace(0, 0.6, 80)
# Adicionar alguns frames estáticos no final (pausa antes de repetir)
frames_radius = np.concatenate([frames_radius, [0.6]*15])

anim = FuncAnimation(fig, update, frames=frames_radius, interval=100)

plt.tight_layout()
plt.show()

# Opcional: Se quiser guardar a animação como GIF, descomente a linha abaixo!
# anim.save('homologia_persistente.gif', writer='pillow', fps=10)