import sympy as sp

# Dimensiones 3D
def vec3(prefix):
    return sp.Matrix(sp.symbols(f'{prefix}1 {prefix}2 {prefix}3'))

# Variables del estado
R = sp.Matrix(sp.symbols('R11 R12 R13 R21 R22 R23 R31 R32 R33')).reshape(3, 3)
omega = vec3('omega')                # Velocidad angular
v_i = vec3('vi')                     # Velocidad del frame i
v_j = vec3('vj')                     # Velocidad del frame j
p = vec3('p')                        # Posición
bg = vec3('bg')                      # Bias giroscópico
g = vec3('g')                        # Gravedad

# Tiempo entre frames
dt = sp.Symbol('dt')

# Estado completo como vector columna (concatenamos todas las variables)
x = sp.Matrix.vstack(
    R.reshape(9, 1),  # R como vector columna
    omega,
    v_i,
    v_j,
    p,
    bg,
    g
)

# Sistema dinámico f(x)
omega_hat = sp.Matrix([
    [0, -omega[2], omega[1]],
    [omega[2], 0, -omega[0]],
    [-omega[1], omega[0], 0]
])
R_dot = sp.MatrixSymbol('R_dot', 3, 3)  # Representación simbólica de R_dot

f = sp.Matrix.vstack(
    (R * omega_hat).reshape(9, 1),
    sp.Matrix.zeros(3, 1),  # dot(omega)
    sp.Matrix.zeros(3, 1),  # dot(vi)
    sp.Matrix.zeros(3, 1),  # dot(vj)
    v_j,                    # dot(p) = vj
    sp.Matrix.zeros(3, 1),  # dot(bg)
    sp.Matrix.zeros(3, 1)   # dot(g)
)

# Medidas
delta_phi_ij = vec3('dphi')
delta_v_ij = vec3('dv')
delta_p_ij = vec3('dp')

# h(x): modelo de medida
# Exp(delta_phi_ij) no se deriva aún, la mantenemos simbólica
Exp_delta_phi = sp.MatrixSymbol('Exp_dphi', 3, 3)
h1 = R.T * sp.MatrixSymbol('Rj', 3, 3) * Exp_delta_phi
h2 = omega + bg
h3 = R.T * (v_j - v_i - g * dt) + delta_v_ij
h4 = R.T * (vec3('pj') - vec3('pi') - v_i * dt - 0.5 * g * dt**2) + delta_p_ij
h5 = vec3('pGPSi') + (vec3('pj') - vec3('pi'))

# Unificamos la salida
h = sp.Matrix.vstack(
    sp.Matrix(h1).reshape(9, 1),
    h2,
    h3,
    h4,
    h5
)

# Derivadas necesarias para matriz de observabilidad
# Primera derivada de Lie: Lf h(x)
grad_h = h.jacobian(x)
Lf0_h = grad_h                # ∇L_f^0 h(x)
Lf1_h = (grad_h * f).jacobian(x)  # ∇L_f^1 h(x)

# Mostrar resultados
print("∇L_f^0 h(x):")
sp.pprint(Lf0_h)

#print("\n∇L_f^1 h(x):")
#sp.pprint(Lf1_h)
