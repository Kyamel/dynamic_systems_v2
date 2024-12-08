from typing import Callable

import numpy as np
from graph import GraphCilinder, GraphTriangle, GraphCicle
from dash import html, dcc, Input, Output
import plotly.graph_objs as go

class DrawableGraphTriangle(GraphTriangle):
    def __init__(self, name: str, type: str, function: Callable[[float], float], function_inverse: Callable[[float], float]) -> None:
        super().__init__(name, type, function, function_inverse)
        
        self.a = 0
        self.b = 0
        self.n = 0
        
        self.slider_min_value = -1
        self.slider_max_value = 1
        self.slider_step = 0.01
        self.slider_marks = {round(i, 1): str(round(i, 1)) for i in np.arange(-1, 1, 0.2)}

        self.slider_iterations_min_value = -32
        self.slider_iterations_max_value = 32
        self.slider_iterations_step = 1
        self.slider_iterations_marks = {i: str(i) for i in range(-32, 33, 4)}

        self.draw_interval = False
        self.interval_size = 0.1


    def layout(self) -> None:
        return html.Div([
            
            # Graph iterations
            html.Div([
                dcc.Graph(
                    id=f'grafico-{self.name}',
                    style={'width': '100%'}
                ),
            ], style={'width': '50%', 'display': 'inline-block'}, className='graph'),

            # Graph hiperplane
            html.Div([
                dcc.Graph(
                    id=f'grafico-hiperplano-{self.name}',
                    style={'width': '100%'}
                ),
            ], style={'width': '50%', 'display': 'inline-block'}, className='graph'),
            
            # Menu
            html.Div([
            dcc.Checklist(
                id=f'options-{self.name}',
                options=[
                    {'label': 'Mostrar iterações pos e neg', 'value': 'both-iterations'},
                    {'label': 'Mostrar pontos hiperplano', 'value': 'show-points-hiperplane'},
                    {'label': 'Mostrar intervalo', 'value': 'show-interval'}
                ],
                value=[],  # Inicialmente, todas as opções estão desmarcadas
                style={'margin-top': '10px', 'margin-bottom': '10px', 'display': 'flex', 'align-items': 'center'},
                labelStyle={'margin-right': '10px'}  # Adiciona margem entre as opções
            ),
            dcc.Input(
                    id=f'interval-size-input-{self.name}',
                    type='number',  # Tipo de entrada numérica para o tamanho do arco
                    placeholder='Tamanho do intervalo...',
                    style={'margin-left': '15px', 'width': '150px'},  # Margem à esquerda para separação e largura do campo
                    value=self.interval_size,
                    step=self.slider_step,
                )
            ], className='graph_menu', style={'display': 'flex', 'align-items': 'center'}),

            # Sliders
            html.Div([
                html.A("Valor inicial: A"),
                dcc.Slider(
                    id=f'slider-a-{self.name}',
                    min=self.slider_min_value,
                    max=self.slider_max_value,
                    step=self.slider_step,
                    value=self.a,
                    marks=self.slider_marks,
                    tooltip={"placement": "bottom", "always_visible": True},
                    className='slider common-slider slider-pontoA',
                    updatemode='drag',
                ),
                html.Div(
                    children=[
                        html.A("Valor inicial: B"),
                        dcc.Slider(
                            id=f'slider-b-{self.name}',
                            min=self.slider_min_value,
                            max=self.slider_max_value,
                            step=self.slider_step,
                            value=self.b,
                            marks=self.slider_marks,
                            tooltip={"placement": "bottom", "always_visible": True},
                            className='slider common-slider slider-pontoB',
                            updatemode='drag',
                        )
                    ],
                    id=f'hide-slider-b-{self.name}'  # Adicione um ID ao container para referência
                ),
                html.A("Número de iterações:"),
                dcc.Slider(
                    id=f'slider-n-{self.name}',
                    min=self.slider_iterations_min_value,
                    max=self.slider_iterations_max_value,
                    step=self.slider_iterations_step,
                    value=self.n,
                    marks=self.slider_iterations_marks,
                    tooltip={"placement": "bottom", "always_visible": True},
                    className='slider common-slider slider-iteracoes',
                    updatemode='drag',
                ),
            ], className='graph_slider'),
        ], style={'width': '100%', 'margin-left': 'auto', 'margin-right': 'auto'})  

    def update(self, xa: float, xb: float, n_iteracoes: int, args='all') -> go.Figure:
        dados = self.generate(xa, xb, n_iteracoes)
        
        if args == 'hiperplane':
            dados_filtrados = [
                dados.get('trace_points_a'),                # trace_points_a
                dados.get('trace_points_b'),                # trace_points_b
                dados.get('trace_points_a_inv'),            # trace_points_a_inv (pode ser None)
                dados.get('trace_points_b_inv'),            # trace_points_b_inv (pode ser None)
                dados.get('trace_triangle'),                # trace_triangle
                dados.get('trace_hiperplane_triangle'),     # trace_hiperplane_triangle
                dados.get('trace_hiperplane_triangle_inv'), # trace_hiperplane_triangle_inv (pode ser None)
                dados.get('trace_interval'),
            ]
        elif args == 'iteration':
            dados_filtrados = [
                dados.get('trace_seg_a'),                   # trace_seg_a
                dados.get('trace_seg_b'),                   # trace_seg_b
                dados.get('trace_seg_a_inv'),               # trace_seg_a_inv (pode ser None)
                dados.get('trace_seg_b_inv'),               # trace_seg_b_inv (pode ser None)
                dados.get('trace_identity_xy'),             # trace_identity_xy
                dados.get('trace_function_triangle'),       # trace_function_triangle
                dados.get('trace_function_triangle_inv'),   # trace_function_triangle_inv (pode ser None)
                dados.get('trace_interval'),                # trace_interval
            ]
        else:
            dados_filtrados = list(dados.values())

        # Remover entradas None (quando as chaves _inv não são retornadas)
        dados_filtrados = [d for d in dados_filtrados if d is not None]

        figura = go.Figure(
            data=dados_filtrados, 
            layout=go.Layout(
                title=f'Gráfico {self.name}',
                xaxis=dict(
                    scaleratio=1,
                    scaleanchor='y',
                ),
                yaxis=dict(
                    scaleratio=1,
                    scaleanchor='x',
                ),
                margin=dict(l=0, r=0, t=40, b=40),
            )
        )                     
        
        return figura

    def callback(self, app) -> None:
        @app.callback(
            [
                Output(f'grafico-{self.name}', 'figure'),
                Output(f'grafico-hiperplano-{self.name}', 'figure'),
                Output(f'hide-slider-b-{self.name}', 'style'),
            ],
            [
                # Sliders
                Input(f'slider-a-{self.name}', 'value'),
                Input(f'slider-b-{self.name}', 'value'),
                Input(f'slider-n-{self.name}', 'value'),
                # Menu
                Input(f'options-{self.name}', 'value'),
                Input(f'interval-size-input-{self.name}', 'value'),
            ],
            prevent_initial_call=False,
        )
        def update_graphs_callback(a, b, n, options, interval_size):
            self.a = a
            self.b = b
            self.n = n
            self.show_points_on_hiperplane = True if 'show-points-hiperplane' in options else False
            self.show_interval = True if 'show-interval' in options else False
            self.show_pos_and_neg_orbits = True if 'both-iterations' in options else False

            if interval_size is not None:
                self.interval_size = interval_size

            if self.show_interval:
                b = a + self.interval_size

            figura_iteration = self.update(a, b, n, args='iteration')
            figura_hiperplane = self.update(a, b, n, args='hiperplane')

            # Condição para o estilo do slider
            slider_style = {'display': 'none'} if self.show_interval else {'display': 'block'}


            return figura_iteration, figura_hiperplane, slider_style



class DrawableGraphCicle(GraphCicle):
    def __init__(self, name: str, type: str, function: Callable[[float], float], function_inverse: Callable[[float], float]) -> None:
        super().__init__(name, type, function, function_inverse)

        self.a = 0
        self.b = 0
        self.n = 0
        
        self.slider_min_value = -2*np.pi
        self.slider_max_value = 2*np.pi
        self.slider_step = 0.01
        # Use np.arange com um valor de término ajustado
        self.slider_marks = {round(i, 1): str(round(i, 1)) for i in np.arange(-6.5, 6.6, 0.5)}

        self.slider_iterations_min_value = -32
        self.slider_iterations_max_value = 32
        self.slider_iterations_step = 1
        self.slider_iterations_marks = {i: str(i) for i in range(-32, 33, 4)}


        self.draw_arc = False
        self.arc_size = 1.0


    def layout(self) -> None:
        return html.Div([
            
            # Graph iterations
            html.Div([
                dcc.Graph(
                    id=f'grafico-{self.name}',
                    style={'width': '100%'}
                ),
            ], style={'width': '50%', 'display': 'inline-block'}, className='graph'),

            # Graph hiperplane
            html.Div([
                dcc.Graph(
                    id=f'grafico-hiperplano-{self.name}',
                    style={'width': '100%'}
                ),
            ], style={'width': '50%', 'display': 'inline-block'}, className='graph'),

            # Menu
            html.Div([
            dcc.Checklist(
                id=f'options-{self.name}',
                options=[
                    {'label': 'Mostrar iterações pos e neg', 'value': 'both-iterations'},
                    {'label': 'Mostrar pontos hiperplano', 'value': 'show-points-hiperplane'},
                    {'label': 'Mostrar arco', 'value': 'show-arc'},
                    {'label': 'Auto incrementar', 'value': 'auto-increment'},
                ],
                value=[],  # Inicialmente, todas as opções estão desmarcadas
                style={'margin-top': '10px', 'margin-bottom': '10px', 'display': 'flex', 'align-items': 'center'},
                labelStyle={'margin-right': '10px'}  # Adiciona margem entre as opções
            ),
            dcc.Input(
                    id=f'arc-size-input-{self.name}',
                    type='number',  # Tipo de entrada numérica para o tamanho do arco
                    placeholder='Tamanho do arco...',
                    style={'margin-left': '15px', 'width': '150px'},  # Margem à esquerda para separação e largura do campo
                    value=self.arc_size,
                    step=self.slider_step,
                ),
            dcc.Input(
                    id=f'auto-increment-input-{self.name}',
                    type='number',  # Tipo de entrada numérica para o tamanho do arco
                    placeholder='Incremento...',
                    style={'margin-left': '15px', 'width': '150px'},  # Margem à esquerda para separação e largura do campo
                    value=self.arc_size,
                    step=self.slider_step,
                )
            ], className='graph_menu', style={'display': 'flex', 'align-items': 'center'}),

            # Auto increment
            dcc.Interval(
                id=f'auto-update-{self.name}',
                interval=1000,  # Intervalo em milissegundos
                max_intervals=0,  # Inicialmente desativado
                disabled=True  # Inicialmente desativado
            ),

            # Sliders
            html.Div([
                html.A("Valor inicial: A"),
                dcc.Slider(
                    id=f'slider-a-{self.name}',
                    min=self.slider_min_value,
                    max=self.slider_max_value,
                    step=self.slider_step,
                    value=self.a,
                    marks=self.slider_marks,
                    tooltip={"placement": "bottom", "always_visible": True},
                    className='slider common-slider slider-pontoA',
                    updatemode='drag',
                ),
                html.Div(
                    children=[
                        html.A("Valor inicial: B"),
                        dcc.Slider(
                            id=f'slider-b-{self.name}',
                            min=self.slider_min_value,
                            max=self.slider_max_value,
                            step=self.slider_step,
                            value=self.b,
                            marks=self.slider_marks,
                            tooltip={"placement": "bottom", "always_visible": True},
                            className='slider common-slider slider-pontoB',
                            updatemode='drag',
                        )
                    ],
                    id=f'hide-slider-b-{self.name}'  # Adicione um ID ao container para referência
                ),
                html.A("Número de iterações:"),
                dcc.Slider(
                    id=f'slider-n-{self.name}',
                    min=self.slider_iterations_min_value,
                    max=self.slider_iterations_max_value,
                    step=self.slider_iterations_step,
                    value=self.n,
                    marks=self.slider_iterations_marks,
                    tooltip={"placement": "bottom", "always_visible": True},
                    className='slider common-slider slider-iteracoes',
                    updatemode='drag',
                ),
            ], className='graph_slider'),
        ], style={'width': '100%', 'margin-left': 'auto', 'margin-right': 'auto'})  

    def update(self, xa: float, xb: float, n_iteracoes: int, args='all') -> go.Figure:
        dados = self.generate(xa, xb, n_iteracoes)

        if args == 'hiperplane':
            dados_filtrados = [
                dados.get('trace_points_a'),           # trace_points_a
                dados.get('trace_points_b'),           # trace_points_b
                dados.get('trace_points_a_inv'),       # trace_points_a_inv (pode ser None)
                dados.get('trace_points_b_inv'),       # trace_points_b_inv (pode ser None)
                dados.get('trace_circle'),             # trace_circle
                dados.get('trace_hiperplane_arc'),     # trace_hiperplane_arc
                dados.get('trace_hiperplane_arc_inv'), # trace_hiperplane_arc_inv (pode ser None)
                dados.get('trace_arc_circle')          # trace_arc_circle
            ]
        elif args == 'iteration':
            dados_filtrados = [
                dados.get('trace_seg_a'),              # trace_seg_a
                dados.get('trace_seg_b'),              # trace_seg_b
                dados.get('trace_seg_a_inv'),          # trace_seg_a_inv (pode ser None)
                dados.get('trace_seg_b_inv'),          # trace_seg_b_inv (pode ser None)
                dados.get('trace_identity_xy'),        # trace_identity_xy
                dados.get('trace_function_arc'),       # trace_function_arc
                dados.get('trace_function_arc_inv'),   # trace_function_arc_inv (pode ser None)
                dados.get('trace_arc')                 # trace_arc
            ]
        else:
            dados_filtrados = list(dados.values())

        # Remover entradas None (quando as chaves _inv não são retornadas)
        dados_filtrados = [d for d in dados_filtrados if d is not None]

        figura = go.Figure(
            data=dados_filtrados, 
            layout=go.Layout(
                title=f'Gráfico {self.name}',
                xaxis=dict(
                    scaleratio=1,
                    scaleanchor='y',
                ),
                yaxis=dict(
                    scaleratio=1,
                    scaleanchor='x',
                ),
                margin=dict(l=0, r=0, t=40, b=40),
            )
        )                     

        return figura


    def callback(self, app) -> None:
        @app.callback(
            [
                Output(f'grafico-{self.name}', 'figure'),
                Output(f'grafico-hiperplano-{self.name}', 'figure'),
                Output(f'hide-slider-b-{self.name}', 'style'),
            ],
            [
                # Siliders
                Input(f'slider-a-{self.name}', 'value'),
                Input(f'slider-b-{self.name}', 'value'),
                Input(f'slider-n-{self.name}', 'value'),
                # Menu
                Input(f'options-{self.name}', 'value'),
                Input(f'arc-size-input-{self.name}', 'value'),
                Input(f'auto-increment-input-{self.name}', 'value'),
            ],
            prevent_initial_call=False,
        )
        def update_graphs_callback(a, b, n, options, arc_size, auto_increment_value):
            self.a = a
            self.b = b
            self.n = n

            self.show_pos_and_neg_orbits = True if 'both-iterations' in options else False
            self.show_points_on_hiperplane = True if 'show-points-hiperplane' in options else False
            self.show_arc = True if 'show-arc' in options else False

            auto_increment = True if 'auto-increment' in options else False

            if auto_increment:
                # Incrementa os valores
                a += auto_increment_value
                b += auto_increment_value

                # Faz o wrap-around para manter os valores dentro do intervalo [0, 2pi)
                a = a % (2 * np.pi)
                b = b % (2 * np.pi)

            if arc_size is not None:
                self.arc_size = arc_size

            if self.show_arc:
                b = a + self.arc_size

            figura_iteration = self.update(a, b, n, args='iteration')
            figura_hiperplane = self.update(a, b, n, args='hiperplane')

            # Condição para o estilo do slider
            slider_style = {'display': 'none'} if self.show_arc else {'display': 'block'}

            return figura_iteration, figura_hiperplane, slider_style



class DrawableGraphCilinder(GraphCilinder):

    def __init__(self, name: str, function_triangle_name: str, function_triangle: Callable[[float], float], function_triangle_inverse: Callable[[float], float], function_arc_name: str, function_arc: Callable[[float], float], function_arc_inverse: Callable[[float], float]):
        super().__init__(name, function_triangle_name, function_triangle, function_triangle_inverse, function_arc_name, function_arc, function_arc_inverse)

        self.a = 0
        self.b = 0
        self.c = 0
        self.n = 0

        self.slider_min_value = -2*np.pi
        self.slider_max_value = 2*np.pi
        self.slider_step = 0.01
        self.slider_marks = {round(i, 1): str(round(i, 1)) for i in np.arange(-6.5, 6.6, 0.5)}

        self.slider_c_min_value = 0
        self.slider_c_max_value = 1
        self.slider_c_step = 0.01
        self.slider_c_marks =  {round(i, 1): str(round(i, 1)) for i in np.arange(0, 1, 0.1)}

         
        self.slider_iterations_min_value = -32
        self.slider_iterations_max_value = 32
        self.slider_iterations_step = 1
        self.slider_iterations_marks = {i: str(i) for i in range(-32, 33, 4)}

        self.draw_arc = False
        self.arc_size = 1.0
        
    def layout(self) -> None:
        return html.Div([

            # Graph iterations
            html.Div([
                dcc.Graph(
                    id=f'grafico-{self.name}',
                    style={'width': '100%'}
                ),
            ], style={'width': '50%', 'display': 'inline-block'}, className='graph'),

            # Graph hiperplane
            html.Div([
                dcc.Graph(
                    id=f'grafico-hiperplano-{self.name}',
                    style={'width': '100%'}
                ),
            ], style={'width': '50%', 'display': 'inline-block'}, className='graph'),
            
            # Menu
            html.Div([
                dcc.Checklist(
                    id=f'both-iterations-{self.name}',
                    options=[
                        {'label': 'Mostrar iterações pos e neg', 'value': 'both-iterations'}
                    ],
                    value=[],
                    style={'margin-top': '10px', 'margin-bottom': '10px'},
                    labelStyle={'margin-right': '10px'}  # Adiciona margem entre as opções
                ),
            ], className='graph_menu'),

            # Sliders
            html.Div([
                html.A("Valor inicial: A"),
                dcc.Slider(
                    id=f'slider-a-{self.name}',
                    min=self.slider_min_value,
                    max=self.slider_max_value,
                    step=self.slider_step,
                    value=self.a,
                    marks=self.slider_marks,
                    tooltip={"placement": "bottom", "always_visible": True},
                    className='slider common-slider slider-pontoA',
                    updatemode='drag',
                ),
                html.A("Valor inicial: B"),
                dcc.Slider(
                    id=f'slider-b-{self.name}',
                    min=self.slider_min_value,
                    max=self.slider_max_value,
                    step=self.slider_step,
                    value=self.b,
                    marks=self.slider_marks,
                    tooltip={"placement": "bottom", "always_visible": True},
                    className='slider common-slider slider-pontoB',
                    updatemode='drag',
                ),
                html.A("Valor inicial: C"),
                dcc.Slider(
                    id=f'slider-c-{self.name}',
                    min=self.slider_c_min_value,
                    max=self.slider_c_max_value,
                    step=self.slider_c_step,
                    value=self.c,
                    marks=self.slider_c_marks,
                    tooltip={"placement": "bottom", "always_visible": True},
                    className='slider common-slider slider-pontoB',
                    updatemode='drag',
                ),
                html.A("Número de iterações:"),
                dcc.Slider(
                    id=f'slider-n-{self.name}',
                    min=self.slider_iterations_min_value,
                    max=self.slider_iterations_max_value,
                    step=self.slider_iterations_step,
                    value=self.n,
                    marks=self.slider_iterations_marks,
                    tooltip={"placement": "bottom", "always_visible": True},
                    className='slider common-slider slider-iteracoes',
                    updatemode='drag',
                ),
            ], className='graph_slider'),
        ], style={'width': '100%', 'margin-left': 'auto', 'margin-right': 'auto'})

    def update(self, xa: float, xb: float,xc: float, n_iteracoes: int, arg = 'all') -> go.Figure:

        dados = self.generate(xa, xb, xc, n_iteracoes)

        if arg == 'hiperplane':
            dados_filtrados = [
                dados['arc'].get('points_a'),           # trace_points_a
                dados['arc'].get('points_b'),           # trace_points_b
                dados['triangle'].get('points_bt'),
                dados['triangle'].get('points_c'),

                dados['arc'].get('points_a_inv'),       # trace_points_a_inv (pode ser None)
                dados['arc'].get('points_b_inv'),       # trace_points_b_inv (pode ser None)
                dados['triangle'].get('points_bt_inv'),
                dados['triangle'].get('points_c_inv'),

                dados['arc'].get('circle'),             # trace_circle
                #dados['arc'].get('arc_circle'),
                dados['triangle'].get('triangle'),
                dados['triangle'].get('identity_zy'),

                dados['hiperplane'].get('points_hiperplane'),
                dados['hiperplane'].get('points_hiperplane_inv'),

                dados['hiperplane'].get('coracao'),
            ]
        elif arg == 'iteration':
            dados_filtrados = [
                dados['arc'].get('seg_a'),              # trace_seg_a
                dados['arc'].get('seg_b'),              # trace_seg_b
                dados['triangle'].get('seg_bt'),
                dados['triangle'].get('seg_c'),

                dados['arc'].get('seg_a_inv'),          # trace_seg_a_inv (pode ser None)
                dados['arc'].get('seg_b_inv'),
                dados['triangle'].get('seg_bt_inv'),
                dados['triangle'].get('seg_c_inv'),

                dados['triangle'].get('function_arc'),       # trace_function_arc
                dados['triangle'].get('function_arc_inv'),   # trace_function_arc_inv (pode ser None)

                dados['arc'].get('circle'),             # trace_circle
                #dados['arc'].get('arc_circle'),
                dados['triangle'].get('triangle'),
                dados['arc'].get('identity_xy'),
                dados['triangle'].get('identity_zy'),

                dados['arc'].get('function_arc'),    # trace_function_circle
                dados['arc'].get('function_arc_inv'),
                dados['triangle'].get('function_t'),
                dados['triangle'].get('function_t_inv'),

            ]
        else:
            dados_filtrados = list(dados['arc'].values()) + list(dados['triangle'].values())

        # Remover entradas None (quando as chaves _inv não são retornadas)
        dados_filtrados = [d for d in dados_filtrados if d is not None]

        # Supondo que dados_filtrados seja uma lista de trace(s) 3D
        figura = go.Figure(
            data=dados_filtrados,
            layout=go.Layout(
                title=f'Gráfico {self.name}',
                scene=dict(
                    xaxis=dict(
                        title='Eixo X',
                        showgrid=True,
                        zeroline=True,
                        showline=True,
                        showspikes=True
                    ),
                    yaxis=dict(
                        title='Eixo Y',
                        showgrid=True,
                        zeroline=True,
                        showline=True,
                        showspikes=True
                    ),
                    zaxis=dict(
                        title='Eixo Z',
                        showgrid=True,
                        zeroline=True,
                        showline=True,
                        showspikes=True
                    ),
                    camera=dict(
                        eye=dict(x=1, y=-2, z=1)  # Troca x e y, movendo y para trás
                    )
                ),
                margin=dict(l=0, r=0, t=40, b=40),
            )
        )

        return figura

    def callback(self, app) -> None:
        @app.callback(
            [
                Output(f'grafico-{self.name}', 'figure'),
                Output(f'grafico-hiperplano-{self.name}', 'figure'),
            ],
            [
                Input(f'slider-a-{self.name}', 'value'),
                Input(f'slider-b-{self.name}', 'value'),
                Input(f'slider-c-{self.name}', 'value'),
                Input(f'slider-n-{self.name}', 'value'),
                Input(f'both-iterations-{self.name}', 'value'),
            ],
            prevent_initial_call=False,
        )
        def update_graph_callback(a, b, c, n, both_iterations):
            self.a = a
            self.b = b
            self.c = c
            self.n = n

            self.show_pos_and_neg_orbits = True if 'both-iterations' in both_iterations else False

            if(self.draw_arc == True):
                b = a + self.arc_size

            figura_iterations = self.update(a, b, c, n, 'iteration')
            figura_hiperplane = self.update(a, b, c, n, 'hiperplane')

            return figura_iterations, figura_hiperplane
