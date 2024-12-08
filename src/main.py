from dash import Dash, html
import numpy as np
from scipy.optimize import fsolve

from drawable import DrawableGraphCilinder, DrawableGraphTriangle, DrawableGraphCicle

app = Dash(__name__, external_stylesheets=['../assets/styles.css'], suppress_callback_exceptions=False)

def f3(x: float) -> float:
    return x + 0.5 * np.sin(x)

def f3_inv(y: float) -> float:
    # Defina a função a ser resolvida: f(x) - y = 0
    equation = lambda x: f3(x) - y
    
    # Adivinhe um valor inicial para a solução
    guess = 0.0

    # Use fsolve para encontrar a solução
    solution = fsolve(equation, guess)
    
    return solution[0]  # Retorna a solução encontrada



#arc1 = DrawableGraphCicle('Arc1', 'senoide', )
arc = DrawableGraphCicle('Arc', 'senoide', f3, f3_inv)
square = DrawableGraphTriangle('square', 'triangle', lambda x: x**2, lambda x: x**0.5)
cilinder = DrawableGraphCilinder('Cilinder', 'square', lambda x: x**2, lambda x: x**0.5,'senoide', f3, f3_inv)

# Adiciona o layout dos graficos ao aplicativo
app.layout = html.Div([
    html.H1("Sistemas Dinâmicos"),
    arc.layout(),
    square.layout(),
    cilinder.layout(),
])

arc.callback(app)
square.callback(app)
cilinder.callback(app)

if __name__ == '__main__':
    app.run_server(debug=True)