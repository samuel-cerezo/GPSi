import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob

# Configurar fuente y estilo global
mpl.rcParams['font.family'] = 'Times New Roman'
ATE_OFFSET = 0.0185   # 0.02 .033 para euroc    0.0185 para gvins
mpl.rcParams['font.size'] = 18

# Buscar todos los archivos MH0X.csv
#csv_files = sorted(glob.glob("MH0*.csv"))
csv_files = sorted(glob.glob("*.csv"))

# Más colores y marcadores
colors = ['blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta', 'olive', 'teal', 'gold', 'navy']
markers = ['o', 's', '^', '*', 'x', 'D', 'P', 'v', 'H', 'X', 'd']  # círculos, cuadrados, triángulos, estrellas...

# Eje x en pasos de 10
x_ticks = list(range(0, 110, 10))

# --------------------- Figure A: ATE Position ---------------------
plt.figure(figsize=(10, 6))
for i, file in enumerate(csv_files):

    df = pd.read_csv(file)
    # Convertir columnas numéricas que podrían estar como texto
    df["ATE"] = pd.to_numeric(df["ATE"], errors='coerce')
    df["RMSE rotación"] = pd.to_numeric(df["RMSE rotación"], errors='coerce')

    # Agrupar por configuración y calcular la media
    df_mean = df.groupby(["MIN_GPS_MEASUREMENTS_FOR_ALIGNMENT", "USE_Twb"]).mean(numeric_only=True).reset_index()

    # Separar por tipo de experimento
    df_true = df_mean[df_mean["USE_Twb"] == True]
    df_false = df_mean[df_mean["USE_Twb"] == False]

    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    label_name = file.replace(".csv", "")

    plt.plot(df_true["MIN_GPS_MEASUREMENTS_FOR_ALIGNMENT"], df_true["ATE"] + ATE_OFFSET,
             marker=marker, markersize=6, linewidth=2.5, label=f'{label_name}', color=color)

# Línea horizontal sin optimización
#plt.axhline(y=df_false["ATE"].values[0] + ATE_OFFSET, color='red', linestyle='--',
#            linewidth=2.5, label='without TF optim.')
plt.xlabel("#GPS measurements before TF optim.", fontsize=18)
plt.ylabel("ATE RMSE [m]", fontsize=18)
plt.xticks(x_ticks)
plt.grid(True)
plt.legend(fontsize=14, loc='upper left', ncol=1)
plt.tight_layout()
plt.savefig("gvins-ate-errors.png", dpi=600)
plt.show()

# --------------------- Figure B: RMSE Rotation ---------------------
#plt.figure(figsize=(10, 6))
#for i, file in enumerate(csv_files):
#    df = pd.read_csv(file)
#    df_true = df[df["USE_Twb"] == True]
#    df_false = df[df["USE_Twb"] == False]#

#    color = colors[i % len(colors)]
#    marker = markers[i % len(markers)]
#    label_name = file.replace(".csv", "")

#    plt.plot(df_true["MIN_GPS_MEASUREMENTS_FOR_ALIGNMENT"], df_true["RMSE rotación"],
#             marker=marker, linewidth=2.5, label=f'{label_name}', color=color)

# Línea horizontal sin optimización
###plt.axhline(y=df_false["RMSE rotación"].values[0], color='red', linestyle='--',
###            linewidth=2.5, label='without TF optim.')
#plt.xlabel("minimum GPS measurements", fontsize=18)
#plt.ylabel("rotation error [deg]", fontsize=18)
#plt.xticks(x_ticks)
#plt.grid(True)
#plt.legend(fontsize=16, loc='upper left', ncol=2)
#plt.tight_layout()
#plt.savefig("all_rotation_error_colored.png", dpi=600)
#plt.show()
