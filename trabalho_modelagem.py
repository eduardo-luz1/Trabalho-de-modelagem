
import numpy as np

pi = np.pi
rho = 1e4

def f(r, h):
    return 350*pi*r**2 + 200*pi*r*h

def g(r, h):
    return 500 - pi*r**2*h

def P(r, h):
    excesso = max(0.0, g(r, h))
    return f(r, h) + rho*excesso**2

# Gradiente analítico da função penalizada
def gradiente_P(r, h):
    dr = 700*pi*r + 200*pi*h
    dh = 200*pi*r

    excesso = g(r, h)

    if excesso > 0:
        dr -= 4*rho*pi*r*h*excesso
        dh -= 2*rho*pi*r**2*excesso

    return dr, dh

# Valores iniciais
r, h = 5.0, 10.0
alpha_inicial = 1e-6
tolerancia = 1e-7
max_iter = 1000000

for i in range(max_iter):
    dr, dh = gradiente_P(r, h)
    norma = np.hypot(dr, dh)

    if norma < tolerancia:
        break

    # Busca um passo que diminua P
    alpha = alpha_inicial
    valor_atual = P(r, h)

    while alpha > 1e-16:
        r_novo = r - alpha*dr
        h_novo = h - alpha*dh

        if r_novo > 0 and h_novo > 0:
            if P(r_novo, h_novo) < valor_atual:
                break

        alpha *= 0.5

    if alpha <= 1e-16:
        break

    r, h = r_novo, h_novo

print(f"Iterações: {i}")
print(f"r = {r:.8f}")
print(f"h = {h:.8f}")
print(f"P(r,h) = {P(r,h):.8f}")
print(f"g(r,h) = {g(r,h):.8f}")
print(f"Volume = {pi*r**2*h:.8f}")
