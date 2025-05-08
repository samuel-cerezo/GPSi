import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configurar fuente y estilo global
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 16

# Leer CSV
df = pd.read_csv('MH01-c.csv')


# Separar los datos
df_false = df[df["USE_Twb"] == False]
df_true = df[df["USE_Twb"] == True]

# Eje x en pasos de 5
x_ticks = list(range(0, 110, 10))

# --------------------- Figure 1: RMSE Position ---------------------
plt.figure(figsize=(8, 6))
plt.plot(df_true["MIN_GPS_MEASUREMENTS_FOR_ALIGNMENT"], df_true["RMSE posición"],
         marker='o', linewidth=2.5, label='with TF optim.', color='blue')
plt.axhline(y=df_false["RMSE posición"].values[0], color='red', linestyle='--',
            linewidth=2.5, label='without TF optim.')
plt.xlabel("Minimum GPS Measurements", fontsize=18)
plt.ylabel("RMSE position [m]", fontsize=18)
plt.xticks(x_ticks)
plt.grid(True)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("figure_rmse_position.png", dpi=600)
plt.show()

# --------------------- Figure 2: RMSE Velocity ---------------------
plt.figure(figsize=(8, 6))
plt.plot(df_true["MIN_GPS_MEASUREMENTS_FOR_ALIGNMENT"], df_true["RMSE velocidad"],
         marker='o', linewidth=2.5, label='with TF optim.', color='green')
plt.axhline(y=df_false["RMSE posición"].values[0], color='red', linestyle='--',
            linewidth=2.5, label='without TF optim.')
plt.xlabel("minimum GPS measurements", fontsize=18)
plt.ylabel("RMSE Velocity [m/s]", fontsize=18)
plt.xticks(x_ticks)
plt.grid(True)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("figure_rmse_velocity.png", dpi=600)
plt.show()

# --------------------- Figure 3: Execution Time ---------------------
plt.figure(figsize=(8, 6))
plt.plot(df_true["MIN_GPS_MEASUREMENTS_FOR_ALIGNMENT"], df_true["Tiempo total de ejecución"],
         marker='o', linewidth=2.5, label='with TF optim.', color='purple')
plt.axhline(y=df_false["RMSE posición"].values[0], color='red', linestyle='--',
            linewidth=2.5, label='without TF optim.')
plt.xlabel("minimum GPS measurements", fontsize=18)
plt.ylabel("execution time [s]", fontsize=18)
plt.xticks(x_ticks)
plt.grid(True)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("figure_execution_time.png", dpi=600)
plt.show()

