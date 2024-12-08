from typing import Callable, Dict, List, Tuple
import values
import plotly.graph_objects as go
import numpy as np
from geometry import Interval, ArcOfCircle, Point

class GraphTriangle:
    def __init__(self, name: str, type: str, function: Callable[[float], float], function_inverse: Callable[[float], float]):
        self.name = name
        self.type = type
        self.function = function
        self.function_inverse = function_inverse
        # Optional
        self.show_pos_and_neg_orbits = False
        self.show_segments = False

        self.show_points = True
        self.show_function = True
        self.show_hiperplane = True
        self.show_points_on_hiperplane = True
        self.show_interval = True

    def iterations_on_triangle(self, x: float, n: int, inv: False) -> Tuple[List[Point], List[Point], List[str], List[str]]:
        seg_x = np.array([x])
        seg_y = np.array([0])
        seg_labels = []
        seg_labels.append('Ponto 0')

        # initial points on triangle
        points_x = np.array([x])
        points_y = np.array([0])
        points_labels = []
        points_labels.append('Ponto 0')
        
        if inv: f = self.function_inverse
        else: f = self.function

        for i in range(abs(n)):
            seg_x = np.append(seg_x, x)
            # iteration
            x = f(x)
            points_x = np.append(points_x, x)
            points_y = np.append(points_y, 0)
            if not inv: points_labels.append(f'Ponto {i+1}')
            else : points_labels.append(f'Ponto {-i-1}')

            seg_y = np.append(seg_y, x)
            if not inv: seg_labels.append(f'Ponto {i+1}')
            else:  seg_labels.append(f'Ponto {-i-1}')
            # y=x
            seg_x = np.append(seg_x, x)
            seg_y = np.append(seg_y, x)
            seg_labels.append(f'Ponto auxiliar x=y')


        points = ([Point(points_x[i], points_y[i]) for i in range(len(points_x))])
        segs = ([Point(seg_x[i], seg_y[i]) for i in range(len(seg_x))])

        return segs, points, seg_labels, points_labels

        
    def generate(self, a: float, b: float, n: int) -> Dict[str, go.Scatter]:
        # Intervalo [0, 1)
        a = a % 1
        b = b % 1
        # Espaço
        X = np.linspace(0.0, 1, 100, endpoint=True)
        Y = np.array([self.function(v) for v in X])
        Y_inv = np.array([self.function_inverse(v) for v in X])
        
        # Calcula as iterações e os traços para as órbitas positiva e negativa
        seg_points_a, points_a, seg_labels_a, points_labels_a = self.iterations_on_triangle(a, n, inv=False)
        seg_points_b, points_b, seg_labels_b, points_labels_b = self.iterations_on_triangle(b, n, inv=False)
        seg_points_a_inv, points_a_inv, seg_labels_a_inv, points_labels_a_inv = self.iterations_on_triangle(a, n, inv=True)
        seg_points_b_inv, points_b_inv, seg_labels_b_inv, points_labels_b_inv = self.iterations_on_triangle(b, n, inv=True)

        interval_space = np.linspace(a, b, 100, endpoint=True)

        trace_dict = {
            'trace_seg_a': go.Scatter(x=[p.x for p in seg_points_a], y=[p.y for p in seg_points_a], mode='lines+markers', name='Segmentos A', line=dict(color=values.col_value_A), text=seg_labels_a),
            'trace_seg_b': go.Scatter(x=[p.x for p in seg_points_b], y=[p.y for p in seg_points_b], mode='lines+markers', name='Segmentos B', line=dict(color=values.col_value_B), text=seg_labels_b),
            'trace_seg_a_inv': go.Scatter(x=[p.x for p in seg_points_a_inv], y=[p.y for p in seg_points_a_inv], mode='lines+markers', name='Segmentos inv A', line=dict(color=values.col_value_A), text=seg_labels_a_inv),
            'trace_seg_b_inv': go.Scatter(x=[p.x for p in seg_points_b_inv], y=[p.y for p in seg_points_b_inv], mode='lines+markers', name='Segmentos inv B', line=dict(color=values.col_value_B), text=seg_labels_b_inv),

            'trace_function_triangle': go.Scatter(x=X, y=Y, mode='lines', name=f'Função: {self.name}', line=dict(color=values.col_function)),
            'trace_function_triangle_inv': go.Scatter(x=X, y=Y_inv, mode='lines', name=f'Função Inversa: {self.name}', line=dict(color=values.col_function)),
            'trace_identity_xy': go.Scatter(x=X, y=X, mode='lines', name='Identidade', line=dict(color=values.col_identity)),
            'trace_triangle': go.Scatter(x=[0, 1, 1, 0], y=[0, 0, 1, 0], mode='lines', name='Triângulo', line=dict(color=values.col_triangle)),

            'trace_hiperplane_triangle': go.Scatter(x=[p.x for p in [Interval(a, b).hiperplane() for a, b in zip(points_a, points_b)]], y=[p.y for p in [Interval(a, b).hiperplane() for a, b in zip(points_a, points_b)]], mode='markers', name='Hiperplano', marker=dict(color=values.col_hiperplane_pos), text=points_labels_b),
            'trace_hiperplane_triangle_inv': go.Scatter(x=[p.x for p in [Interval(a, b).hiperplane() for a, b in zip(points_a_inv, points_b_inv)]], y=[p.y for p in [Interval(a, b).hiperplane() for a, b in zip(points_a_inv, points_b_inv)]], mode='markers', name='Hiperplano inv', marker=dict(color=values.col_hiperplane_neg), text=points_labels_b_inv),
        }

        if self.show_interval:
            trace_dict.update({
                'trace_interval': go.Scatter(x=interval_space, y=np.zeros_like(interval_space), mode='lines', name='Intervalo', line=dict(color=values.col_interval)),
            })

        if self.show_points_on_hiperplane:
            trace_dict.update({
                'trace_points_a': go.Scatter(x=[p.x for p in points_a], y=[p.y for p in points_a], mode='markers', name='Pontos A', marker=dict(color=values.col_value_A), text=points_labels_a),
                'trace_points_b': go.Scatter(x=[p.x for p in points_b], y=[p.y for p in points_b], mode='markers', name='Pontos B', marker=dict(color=values.col_value_B), text=points_labels_b),
                'trace_points_a_inv': go.Scatter(x=[p.x for p in points_a_inv], y=[p.y for p in points_a_inv], mode='markers', name='Pontos inv A', marker=dict(color=values.col_value_A), text=points_labels_a_inv),
                'trace_points_b_inv': go.Scatter(x=[p.x for p in points_b_inv], y=[p.y for p in points_b_inv], mode='markers', name='Pontos inv B', marker=dict(color=values.col_value_B), text=points_labels_b_inv),
            })


        if self.show_pos_and_neg_orbits:
            return trace_dict
        else:
            if n >= 0:
                return {key: trace_dict.get(key) for key in ['trace_seg_a', 'trace_seg_b', 'trace_points_a', 'trace_points_b', 'trace_identity_xy', 'trace_triangle', 'trace_function_triangle', 'trace_hiperplane_triangle', 'trace_interval']}
            else:
                return {key: trace_dict.get(key) for key in ['trace_seg_a_inv', 'trace_seg_b_inv', 'trace_points_a_inv', 'trace_points_b_inv', 'trace_identity_xy', 'trace_triangle', 'trace_function_triangle_inv', 'trace_hiperplane_triangle_inv', 'trace_interval']}


class GraphCicle:
    def __init__(self, name: str, type: str, function: Callable[[float], float], function_inverse: Callable[[float], float]):
        self.name = name
        self.type = type
        self.function = function
        self.function_inv = function_inverse
        # Optional
        self.show_pos_and_neg_orbits = False
        self.show_segments = False

        self.show_points = True
        self.show_function = True
        self.show_hiperplane = True
        self.show_points_on_hiperplane = True
        self.show_arc = True

    def iterations_on_cicle(self, x: float, n: int, inv: False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, list]:
        seg_x = np.array([x])
        seg_y = np.array([0])
        seg_labels = []
        seg_labels.append('Ponto 0')

        # initial points on circle
        points_x = np.array([np.cos(x)])
        points_y = np.array([np.sin(x)])
        points_labels = []
        points_labels.append('Ponto 0')
        
        if inv : f = self.function_inv
        else : f = self.function

        for i in range(abs(n)):
            seg_x = np.append(seg_x, x)
            # iteration
            x = f(x)
            # points on circle after iteration
            points_x = np.append(points_x, np.cos(x))
            points_y = np.append(points_y, np.sin(x))
            if not inv: points_labels.append(f'Ponto {i+1}')
            else : points_labels.append(f'Ponto {-i-1}')

            seg_y = np.append(seg_y, x)
            if not inv: seg_labels.append(f'Ponto {i+1}')
            else:  seg_labels.append(f'Ponto {-i-1}')
            # x=y
            seg_x = np.append(seg_x, x)
            seg_y = np.append(seg_y, x)
            seg_labels.append(f'Ponto auxiliar x=y')

        

        points = ([Point(points_x[i], points_y[i]) for i in range(len(points_x))])
        segs = ([Point(seg_x[i], seg_y[i]) for i in range(len(seg_x))])

        return segs, points, seg_labels, points_labels

    def generate(self, a: float, b: float, n: int) -> Dict[str, go.Scatter]:
        # interval [0, 2pi)
        a = a % (2 * np.pi)
        b = b % (2 * np.pi)

        # espaço arco
        X_arc = np.linspace(0.0, 2*np.pi, 100, endpoint=True)
        Y_arc = np.array([self.function(v) for v in X_arc])
        Y_arc_inv = np.array([self.function_inv(v) for v in X_arc])

        seg_points_a, points_a, seg_labels_a, points_labels_a = self.iterations_on_cicle(a, n, inv=False)
        seg_points_b, points_b, seg_labels_b, points_labels_b = self.iterations_on_cicle(b, n, inv=False)

        seg_points_a_inv, points_a_inv, seg_labels_a_inv, points_labels_a_inv = self.iterations_on_cicle(a, n, inv=True)
        seg_points_b_inv, points_b_inv, seg_labels_b_inv, points_labels_b_inv = self.iterations_on_cicle(b, n, inv=True)

        arc_space = np.linspace(a, b, 100, endpoint=True)

        trace_dict = {
            'trace_seg_a': go.Scatter(x=[p.x for p in seg_points_a], y=[p.y for p in seg_points_a], mode='lines+markers', name='Segmentos A', line=dict(color=values.col_value_A), text=seg_labels_a),
            'trace_seg_b': go.Scatter(x=[p.x for p in seg_points_b], y=[p.y for p in seg_points_b], mode='lines+markers', name='Segmentos B', line=dict(color=values.col_value_B), text=seg_labels_b),
            'trace_seg_a_inv': go.Scatter(x=[p.x for p in seg_points_a_inv], y=[p.y for p in seg_points_a_inv], mode='lines+markers', name='Segmentos inv A', line=dict(color=values.col_value_A), text=seg_labels_a_inv),
            'trace_seg_b_inv': go.Scatter(x=[p.x for p in seg_points_b_inv], y=[p.y for p in seg_points_b_inv], mode='lines+markers', name='Segmentos inv B', line=dict(color=values.col_value_B), text=seg_labels_b_inv),

            'trace_function_arc': go.Scatter(x=X_arc, y=Y_arc, mode='lines', name=f'Função: {self.type}', line=dict(color=values.col_function)),
            'trace_function_arc_inv': go.Scatter(x=X_arc, y=Y_arc_inv, mode='lines', name=f'Função Inversa: {self.type}', line=dict(color=values.col_function)),

            'trace_identity_xy': go.Scatter(x=X_arc, y=X_arc, mode='lines', name='Identidade', line=dict(color=values.col_identity)),
            'trace_circle': go.Scatter(x=np.cos(X_arc), y=np.sin(X_arc), mode='lines', name='Circulo', line=dict(color=values.col_circle)),

        }

        arc = [ArcOfCircle(a, Point(0, 0), b) for a, b in zip(points_a, points_b)]
        arc_inv = [ArcOfCircle(a, Point(0, 0), b) for a, b in zip(points_a_inv, points_b_inv)]

        hiperplane_arc = [p.hiperplane() for p in arc]
        hiperplane_arc_inv = [p.hiperplane() for p in arc_inv]

        trace_dict['trace_hiperplane_arc'] = go.Scatter(x=[p.x for p in hiperplane_arc], y=[p.y for p in hiperplane_arc], mode='markers', name='Hiperplano', marker=dict(color=values.col_hiperplane_pos), text=points_labels_a)
        trace_dict['trace_hiperplane_arc_inv'] = go.Scatter(x=[p.x for p in hiperplane_arc_inv], y=[p.y for p in hiperplane_arc_inv], mode='markers', name='Hiperplano inv', marker=dict(color=values.col_hiperplane_neg), text=points_labels_a_inv)

        if self.show_points_on_hiperplane:
            trace_dict.update({
                'trace_points_a': go.Scatter(x=[p.x for p in points_a], y=[p.y for p in points_a], mode='markers', name='Pontos A', marker=dict(color=values.col_value_A), text=points_labels_a),
                'trace_points_b': go.Scatter(x=[p.x for p in points_b], y=[p.y for p in points_b], mode='markers', name='Pontos B', marker=dict(color=values.col_value_B), text=points_labels_b),
                'trace_points_a_inv': go.Scatter(x=[p.x for p in points_a_inv], y=[p.y for p in points_a_inv], mode='markers', name='Pontos inv A', marker=dict(color=values.col_value_A), text=points_labels_a_inv),
                'trace_points_b_inv': go.Scatter(x=[p.x for p in points_b_inv], y=[p.y for p in points_b_inv], mode='markers', name='Pontos inv B', marker=dict(color=values.col_value_B), text=points_labels_b_inv),
            })

        if self.show_arc:
            trace_dict.update({
                'trace_arc': go.Scatter(x=arc_space, y=np.zeros_like(arc_space), mode='lines', name='Arco', line=dict(color=values.col_arc)),
                'trace_arc_circle': go.Scatter(x=np.cos(arc_space), y=np.sin(arc_space), mode='lines', name='Arco', line=dict(color=values.col_arc)),
            })

        if self.show_pos_and_neg_orbits:
            return trace_dict
        else:
            if n >= 0:
                return {key: trace_dict.get(key) for key in ['trace_seg_a', 'trace_seg_b', 'trace_points_a', 'trace_points_b', 'trace_identity_xy', 'trace_circle', 'trace_function_arc', 'trace_hiperplane_arc', 'trace_arc', 'trace_arc_circle']}
            else:
                return {key: trace_dict.get(key) for key in ['trace_seg_a_inv', 'trace_seg_b_inv', 'trace_points_a_inv', 'trace_points_b_inv', 'trace_identity_xy', 'trace_circle', 'trace_function_arc_inv', 'trace_hiperplane_arc_inv', 'trace_arc', 'trace_arc_circle']}


class GraphCilinder:
    def __init__(self, name: str, function_triangle_name: str, function_triangle: Callable[[float], float], function_triangle_inverse: Callable[[float], float], function_arc_name: str, function_arc: Callable[[float], float], function_arc_inverse: Callable[[float], float]):
        self.name = name
        self.function_triangle_name = function_triangle_name
        self.function_arc_name = function_arc_name

        self.function_triangle = function_triangle
        self.function_triangle_inv = function_triangle_inverse
        self.function_arc = function_arc
        self.function_arc_inv = function_arc_inverse

        # Optional
        self.show_pos_and_neg_orbits = False
        self.show_segments = False

        self.show_points = True
        self.show_function = True
        self.show_hiperplane = True

    def iterations_on_triangle(self, x: float, n: int, inv: False) -> Tuple[List[Point], List[Point], List[str], List[str]]:
        # Intervalo [0, 1)

        seg_x = np.array([x])
        seg_y = np.array([0])
        seg_labels = []
        seg_labels.append('Ponto 0')

        # initial points on triangle
        points_x = np.array([x])
        points_y = np.array([0])
        points_labels = []
        points_labels.append('Ponto 0')
        
        if inv: f = self.function_triangle_inv
        else: f = self.function_triangle

        for i in range(abs(n)):
            seg_x = np.append(seg_x, x)
            # iteration
            x = f(x)
            points_x = np.append(points_x, x)
            points_y = np.append(points_y, 0)
            if not inv: points_labels.append(f'Ponto {i+1}')
            else : points_labels.append(f'Ponto {-i-1}')

            seg_y = np.append(seg_y, x)
            if not inv: seg_labels.append(f'Ponto {i+1}')
            else:  seg_labels.append(f'Ponto {-i-1}')
            # y=x
            seg_x = np.append(seg_x, x)
            seg_y = np.append(seg_y, x)
            seg_labels.append(f'Ponto auxiliar x=y')


        points = ([Point(points_x[i], points_y[i]) for i in range(len(points_x))])
        segs = ([Point(seg_x[i], seg_y[i]) for i in range(len(seg_x))])

        return segs, points, seg_labels, points_labels

    def iterations_on_arc(self, x: float, n: int, inv: False) -> Tuple[List[Point], List[Point], List[str], List[str]]:
        # Interval [0, 2pi)
        x = x % (2 * np.pi)

        seg_x = np.array([x])
        seg_y = np.array([0])
        seg_labels = []
        seg_labels.append('Ponto 0')

        # initial points on circle
        points_x = np.array([np.cos(x)])
        points_y = np.array([np.sin(x)])
        points_labels = []
        points_labels.append('Ponto 0')
        
        if inv : f = self.function_arc_inv
        else : f = self.function_arc

        for i in range(abs(n)):
            seg_x = np.append(seg_x, x)
            # iteration
            x = f(x)
            # points on circle after iteration
            points_x = np.append(points_x, np.cos(x))
            points_y = np.append(points_y, np.sin(x))
            if not inv: points_labels.append(f'Ponto {i+1}')
            else : points_labels.append(f'Ponto {-i-1}')

            seg_y = np.append(seg_y, x)
            if not inv: seg_labels.append(f'Ponto {i+1}')
            else:  seg_labels.append(f'Ponto {-i-1}')
            # x=y
            seg_x = np.append(seg_x, x)
            seg_y = np.append(seg_y, x)
            seg_labels.append(f'Ponto auxiliar x=y')

        points = ([Point(points_x[i], points_y[i]) for i in range(len(points_x))])
        segs = ([Point(seg_x[i], seg_y[i]) for i in range(len(seg_x))])

        return segs, points, seg_labels, points_labels

    def hiperplane_trasform(self, a: float, b: float) -> Tuple[float, float, float]:
        px = -a - 1 # a + 1
        py = 0
        pz = b - a
        return px, py, pz

    def generate(self, a: float, b: float, c: float, n: int) -> Dict[str, Dict[str, go.Scatter3d]]:
        # Interval [0, 1)
        c = c % 1
        # espaço trianglulo
        X_t = np.linspace(0.0, 1, 100, endpoint=True)
        Y_t = np.array([self.function_triangle(v) for v in X_t])
        Y_t_inv = np.array([self.function_triangle_inv(v) for v in X_t])

        # espaço arco
        X_arc = np.linspace(0.0, 2*np.pi, 100, endpoint=True)
        Y_arc = np.array([self.function_arc(v) for v in X_arc])
        Y_arc_inv = np.array([self.function_arc_inv(v) for v in X_arc])

        # Calculando os pontos e segmentos
        seg_points_a, points_a, seg_labels_a, points_labels_a = self.iterations_on_arc(a, n, inv=False)
        seg_points_b, points_b, seg_labels_b, points_labels_b = self.iterations_on_arc(b, n, inv=False)

        # definir bt por partes
        bt = bt = points_b[0].x
        hiperplane_defined = False
        if a < b:
            if a < np.pi and b > np.pi:
                hiperplane_defined = True
        else:
            if not (a > np.pi and b < np.pi):
                hiperplane_defined = True

        seg_points_bt, points_bt, seg_labels_bt, points_labels_bt = self.iterations_on_triangle(b/(2*np.pi), n, inv=False)
        seg_points_c, points_c, seg_labels_c, points_labels_c = self.iterations_on_triangle(c, n, inv=False)

        # Calculando os inversos
        seg_points_a_inv, points_a_inv, seg_labels_a_inv, points_labels_a_inv = self.iterations_on_arc(a, n, inv=True)
        seg_points_b_inv, points_b_inv, seg_labels_b_inv, points_labels_b_inv = self.iterations_on_arc(b, n, inv=True)
        seg_points_bt_inv, points_bt_inv, seg_labels_bt_inv, points_labels_bt_inv = self.iterations_on_triangle(bt, n, inv=True)
        seg_points_c_inv, points_c_inv, seg_labels_c_inv, points_labels_c_inv = self.iterations_on_triangle(c, n, inv=True)

        triangle_space_y = [-1, -2, -1, -1]
        triangle_space_z = [0, 0, 1, 0]

        arc = [ArcOfCircle(a, Point(0, 0), b) for a, b in zip(points_a, points_b)]
        arc_inv = [ArcOfCircle(a, Point(0, 0), b) for a, b in zip(points_a_inv, points_b_inv)]

        hiperplane_arc = [p.hiperplane() for p in arc]
        hiperplane_arc_inv = [p.hiperplane() for p in arc_inv]

        hiperplane_triangle_x = []
        hiperplane_triangle_y = []
        hiperplane_triangle_z = []
        for bt, c in zip(points_bt, points_c):

            b, b, z = self.hiperplane_trasform(bt.x, c.x)
            hiperplane_triangle_x.append(b)
            hiperplane_triangle_y.append(b)
            hiperplane_triangle_z.append(z)

        coracao_points_x = []
        coracao_points_y = []
        # Cria o intervalo crescente
        b_values = np.linspace(0, 2 * np.pi, 50, endpoint = False)
        # Inverte o intervalo
        b_values_fliped = np.linspace(2 * np.pi, 0, 50, endpoint=False)

        for a in b_values:
            pa = Point(np.cos(a), np.sin(a))
            coracao = ArcOfCircle(pa, Point(0, 0), Point(-1, 0))
            point = coracao.hiperplane()
            coracao_points_x.append(point.x)
            coracao_points_y.append(point.y)

        for b in b_values_fliped:
            pb = Point(np.cos(b), np.sin(b))
            coracao = ArcOfCircle(Point(-1, 0), Point(0, 0), pb)
            point = coracao.hiperplane()
            coracao_points_x.append(point.x)
            coracao_points_y.append(point.y)


        # Duplicar os pontos para as bases (z=0 para a base inferior e z=1 para a base superior)
        x = coracao_points_x + coracao_points_x
        y = coracao_points_y + coracao_points_y
        z = [0] * len(coracao_points_x) + [1] * len(coracao_points_x)

        # Definir as faces (triângulos) da superfície lateral
        faces = []
        n = len(coracao_points_x)

        # Faces laterais
        for i in range(n - 1):
            faces.append([i, i + 1, i + 1 + n])
            faces.append([i, i + 1 + n, i + n])

        # Adicionar a face que conecta o último ponto com o primeiro ponto (não será visível se a malha estiver bem definida)
        #faces.append([n - 1, 0, n])
        #aces.append([n - 1, n, n + n - 1])

        # Converter faces para arrays numpy
        i, j, k = np.array(faces).T


        #hiperplane_triangle = [Interval(bt, c).hiperplane() for bt, c in zip(points_bt, points_c)]],

        # Criando os traces
        traces = {
            'arc': {
                'seg_a': go.Scatter3d(x=[p.x for p in seg_points_a], y=[p.y for p in seg_points_a], z=np.zeros_like(seg_points_a), mode='lines', name='Segmentos A', line=dict(color=values.col_value_A), text=seg_labels_a),
                'seg_b': go.Scatter3d(x=[p.x for p in seg_points_b], y=[p.y for p in seg_points_b], z=np.zeros_like(seg_points_b), mode='lines', name='Segmentos B', line=dict(color=values.col_value_B), text=seg_labels_b),
                'points_a': go.Scatter3d(x=[p.x for p in points_a], y=[p.y for p in points_a], z=np.zeros_like(points_a), mode='markers', name='Pontos A', marker=dict(color=values.col_value_A), text=points_labels_a),
                'points_b': go.Scatter3d(x=[p.x for p in points_b], y=[p.y for p in points_b], z=np.zeros_like(points_b), mode='markers', name='Pontos B', marker=dict(color=values.col_value_B), text=points_labels_b),

                'seg_a_inv': go.Scatter3d(x=[p.x for p in seg_points_a_inv], y=[p.y for p in seg_points_a_inv], z=np.zeros_like(seg_points_a_inv), mode='lines', name='Segmentos inv A', line=dict(color=values.col_value_A), text=seg_labels_a_inv),
                'seg_b_inv': go.Scatter3d(x=[p.x for p in seg_points_b_inv], y=[p.y for p in seg_points_b_inv], z=np.zeros_like(seg_points_b_inv), mode='lines', name='Segmentos inv B', line=dict(color=values.col_value_B), text=seg_labels_b_inv),
                'points_a_inv': go.Scatter3d(x=[p.x for p in points_a_inv], y=[p.y for p in points_a_inv], z=np.zeros_like(points_a_inv), mode='markers', name='Pontos inv A', marker=dict(color=values.col_value_A), text=points_labels_a_inv),
                'points_b_inv': go.Scatter3d(x=[p.x for p in points_b_inv], y=[p.y for p in points_b_inv], z=np.zeros_like(points_b_inv), mode='markers', name='Pontos inv B', marker=dict(color=values.col_value_B), text=points_labels_b_inv),

                'function_arc': go.Scatter3d(x=X_arc, y=Y_arc, z=np.zeros_like(X_arc), mode='lines', name=f'Função: {self.function_arc_name}', line=dict(color=values.col_function)),
                'function_arc_inv': go.Scatter3d(x=X_arc, y=Y_arc_inv, z=np.zeros_like(X_arc), mode='lines', name=f'Função Inversa: {self.function_arc_name}', line=dict(color=values.col_function)),

                'identity_xy': go.Scatter3d(x=X_arc, y=X_arc, z=np.zeros_like(X_arc), mode='lines', name='Identidade x=y', line=dict(color=values.col_identity)),
                'circle': go.Scatter3d(x=np.cos(X_arc), y=np.sin(X_arc), z=np.zeros_like(X_arc), mode='lines', name='Circulo', line=dict(color=values.col_circle)),
            },
            'triangle': {
                #'seg_bt': go.Scatter3d(x=[p.x-2 for p in seg_points_bt], y=np.zeros_like(seg_points_bt), z=[p.y for p in seg_points_bt], mode='lines', name="Segmentos B'", line=dict(color=values.col_value_B), text=seg_labels_bt),
                'seg_c': go.Scatter3d(x=[p.x-2 for p in seg_points_c], y=np.zeros_like(seg_points_c), z=[p.y for p in seg_points_c], mode='lines', name='Segmentos C', line=dict(color=values.col_value_C), text=seg_labels_c),
                #'points_bt': go.Scatter3d(x=[p.x for p in points_bt], y=np.zeros_like(points_bt), z=[p.y for p in points_bt], mode='markers', name="Pontos B'", marker=dict(color=values.col_value_B), text=points_labels_bt),
                'points_c': go.Scatter3d(z=[p.x for p in points_c], y=np.zeros_like(points_c), x=[p.y-1 for p in points_c], mode='markers', name='Pontos C', marker=dict(color=values.col_value_C), text=points_labels_c),

                #'seg_bt_inv': go.Scatter3d(x=[p.x-2 for p in seg_points_bt_inv], y=np.zeros_like(seg_points_bt_inv), z=[p.y for p in seg_points_bt_inv], mode='lines', name='Segmentos inv B', line=dict(color=values.col_value_B), text=seg_labels_bt_inv),
                'seg_c_inv': go.Scatter3d(z=[p.x-2 for p in seg_points_c_inv], y=np.zeros_like(seg_points_c_inv), x=[p.y for p in seg_points_c_inv], mode='lines', name='Segmentos inv C', line=dict(color=values.col_value_C), text=seg_labels_c_inv),
                #'points_bt_inv': go.Scatter3d(x=[p.x for p in points_bt_inv], y=np.zeros_like(points_bt_inv), z=[p.y for p in points_bt_inv], mode='markers', name='Pontos inv B', marker=dict(color=values.col_value_B), text=points_labels_bt_inv),
                'points_c_inv': go.Scatter3d(z=[p.x for p in points_c_inv], y=np.zeros_like(points_c_inv), x=[p.y-1 for p in points_c_inv], mode='markers', name='Pontos inv C', marker=dict(color=values.col_value_C), text=points_labels_c_inv),

                'function_t': go.Scatter3d(y=np.zeros_like(X_t), x=X_t-2, z=Y_t, mode='lines', name=f'Função: {self.function_triangle_name}', line=dict(color=values.col_function)),
                'function_t_inv': go.Scatter3d(y=np.zeros_like(X_t), x=X_t-2, z=Y_t_inv, mode='lines', name=f'Função Inversa: {self.function_triangle_name}', line=dict(color=values.col_function)),

                'identity_zy': go.Scatter3d(y=np.zeros_like(X_t), x=X_t-2, z=X_t, mode='lines', name='Identidade z=y', line=dict(color=values.col_identity)),
                'triangle': go.Scatter3d(y=np.zeros_like(triangle_space_y), x=triangle_space_y, z=triangle_space_z, mode='lines', name='Triangulo', line=dict(color=values.col_triangle))
            },
            'hiperplane': {
                'points_hiperplane': go.Scatter3d(x=[p.x for p in hiperplane_arc], y=[p.y for p in hiperplane_arc], z=[p.x for p in points_c], mode='markers', name='Hiperplano', marker=dict(color=values.col_hiperplane_pos)),
                'coracao': go.Mesh3d(   x=x, y=y, z=z,
                                        i=i, j=j, k=k,
                                        opacity=0.5,
                                        color='cyan',
                                        name='coracao',
                                    ),

            },
        }

        if self.show_pos_and_neg_orbits:
            return traces

        # Condicional para mostrar todos os traces
        if self.show_pos_and_neg_orbits:
            return traces

        # Filtragem condicional com base em `n`

        
        filtered_traces = {
            'arc': {
                k: v for k, v in traces['arc'].items()
                if (n >= 0 and '_inv' not in k) or (n < 0 and ('_inv' in k or k in ['identity_xy', 'circle']))
            },
            'triangle': {
                k: v for k, v in traces['triangle'].items()
                if (n >= 0 and '_inv' not in k) or (n < 0 and ('_inv' in k or k in ['identity_zy', 'triangle']))
            },
            'hiperplane': {
                k: v for k, v in traces['hiperplane'].items()
                if (n >= 0 and '_inv' not in k) or (n < 0 and ('_inv' in k or k in ['identity_zy', 'triangle']))
            },
        }

        return filtered_traces

