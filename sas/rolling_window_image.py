import matplotlib.pyplot as plt
import numpy as np

def plot_time_series_cv(total_time_steps=12, min_train_size=4, forecast_horizon=2):
    """
    Generates a Time Series Cross-Validation (Expanding Window) diagram
    with non-overlapping test sets (step size = forecast horizon).
    """
    # CORRECCIÓN 1: Calculamos el número de iteraciones usando división entera (//)
    # Ya que saltamos de 2 en 2 (o el valor que tenga forecast_horizon)
    num_folds = (total_time_steps - min_train_size) // forecast_horizon

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors
    color_train = '#1f77b4'  # Blue
    color_test = '#ff7f0e'   # Orange
    color_unused = '#d3d3d3' # Light Grey

    for i in range(num_folds):
        # Y coordinate for this fold (drawing from top to bottom)
        y = num_folds - i

        # CORRECCIÓN 2: El entrenamiento crece absorbiendo el bloque entero de test
        current_train_size = min_train_size + (i * forecast_horizon)
        
        train_x = np.arange(1, current_train_size + 1)
        test_x = np.arange(current_train_size + 1, current_train_size + 1 + forecast_horizon)
        unused_x = np.arange(current_train_size + 1 + forecast_horizon, total_time_steps + 1)

        # Plot the timeline connecting line
        ax.plot(np.arange(1, total_time_steps + 1), [y]*total_time_steps, 
                color=color_unused, zorder=1, linewidth=1.5)

        # Plot training data
        ax.scatter(train_x, [y]*len(train_x), color=color_train, s=120, zorder=2, label='Entrenamiento' if i==0 else "")

        # Plot testing data
        ax.scatter(test_x, [y]*len(test_x), color=color_test, s=120, zorder=2, label='Validación' if i==0 else "")

        # Plot unused future data
        if len(unused_x) > 0:
            ax.scatter(unused_x, [y]*len(unused_x), color=color_unused, s=120, zorder=2, label='No utilizado' if i==0 else "")

    # Formatting and styling
    ax.set_yticks(np.arange(1, num_folds + 1))
    ax.set_yticklabels([f"Iteración {num_folds - i}" for i in range(num_folds)])
    ax.set_xticks(np.arange(1, total_time_steps + 1))
    
    ax.set_xlabel("Tiempo", fontsize=12, fontweight='bold')
    ax.set_title("Validación Cruzada en Series Temporales", 
                 fontsize=14, fontweight='bold', pad=20)

    # Clean up the axes spines for a modern look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0) # Hide Y axis ticks

    # Add legend at the top right
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1.15), ncol=3, frameon=False)

    plt.tight_layout()
    
    # Save the figure with high DPI for your LaTeX document
    plt.savefig("time_series_cv_non_overlapping.png", dpi=300, bbox_inches='tight')
    plt.show()

# Execute the function (Aumenté total_time_steps a 12 para que se vean 4 iteraciones limpias)
plot_time_series_cv(total_time_steps=12, min_train_size=4, forecast_horizon=2)