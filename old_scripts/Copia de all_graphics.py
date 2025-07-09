import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob

# Configurar fuente y estilo global
mpl.rcParams['font.family'] = 'Times New Roman'
ATE_OFFSET = 0.02
mpl.rcParams['font.size'] = 16

# Buscar todos los archivos MH0X.csv
csv_files = sorted(glob.glob("MH0*.csv"))

# Más colores y marcadores
colors = ['blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta', 'olive', 'teal', 'gold', 'navy']
markers = ['o', 's', '^', '*', 'x', 'D', 'P', 'v', 'H', 'X', 'd']  # círculos, cuadrados, triángulos, estrellas...

# Eje x en pasos de 10
x_ticks = list(range(0, 110, 10))

# --------------------- Figure A: ATE Position ---------------------
plt.figure(figsize=(10, 6))
for i, file in enumerate(csv_files):
    df = pd.read_csv(file)
    df_true = df[df["USE_Twb"] == True]
    df_false = df[df["USE_Twb"] == False]

    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    label_name = file.replace(".csv", "")

    plt.plot(df_true["MIN_GPS_MEASUREMENTS_FOR_ALIGNMENT"], df_true["ATE"] + ATE_OFFSET,
             marker=marker, linewidth=2.5, label=f'{label_name}', color=color)

# Línea horizontal sin optimización
plt.axhline(y=df_false["ATE"].values[0] + ATE_OFFSET, color='red', linestyle='--',
            linewidth=2.5, label='without TF optim.')
plt.xlabel("minimum GPS measurements", fontsize=18)
plt.ylabel("translation error [m]", fontsize=18)
plt.xticks(x_ticks)
plt.grid(True)
plt.legend(fontsize=13, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
plt.tight_layout()
plt.savefig("all_position_error_colored.png", dpi=600)
plt.show()

# --------------------- Figure B: RMSE Rotation ---------------------
plt.figure(figsize=(10, 6))
for i, file in enumerate(csv_files):
    df = pd.read_csv(file)
    df_true = df[df["USE_Twb"] == True]
    df_false = df[df["USE_Twb"] == False]

    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    label_name = file.replace(".csv", "")

    plt.plot(df_true["MIN_GPS_MEASUREMENTS_FOR_ALIGNMENT"], df_true["RMSE rotación"],
             marker=marker, linewidth=2.5, label=f'{label_name}', color=color)

# Línea horizontal sin optimización
plt.axhline(y=df_false["RMSE rotación"].values[0], color='red', linestyle='--',
            linewidth=2.5, label='without TF optim.')
plt.xlabel("minimum GPS measurements", fontsize=18)
plt.ylabel("rotation error [deg]", fontsize=18)
plt.xticks(x_ticks)
plt.grid(True)
plt.legend(fontsize=13, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
plt.tight_layout()
plt.savefig("all_rotation_error_colored.png", dpi=600)
plt.show()
