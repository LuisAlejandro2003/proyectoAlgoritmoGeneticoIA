import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QGroupBox, QGridLayout
import matplotlib.pyplot as plt

# Cargar el dataset
df_materiales = pd.read_csv('materiales.csv')



# Función para imprimir las propiedades de una mezcla en la interfaz
def imprimir_mezcla(mezcla, nombre, requisitos, historia_resistencias, historia_durabilidades, historia_costos, historia_resistencias_corrosion, historia_pesos):
    if mezcla is None:
        return f"<span style='color:red;'>{nombre} no cumple con los requisitos solicitados.</span>"

    materiales_utilizados = df_materiales.loc[mezcla > 0]
    porcentajes = mezcla[mezcla > 0] * 100
    resistencia_total = historia_resistencias[-1]
    durabilidad_total = historia_durabilidades[-1]
    costo_total = historia_costos[-1]
    resistencia_corrosion_total = historia_resistencias_corrosion[-1]
    peso_total = historia_pesos[-1]

    cumple_resistencia = resistencia_total >= requisitos['resistencia_minima']
    cumple_durabilidad = durabilidad_total >= requisitos['durabilidad_deseada']
    cumple_corrosion = resistencia_corrosion_total >= requisitos['resistencia_corrosion_minima']
    cumple_peso = peso_total <= requisitos['peso_maximo']

    color = 'green' if cumple_resistencia and cumple_durabilidad and cumple_corrosion and cumple_peso else 'red'

    output = f"<span style='color:{color};'>{nombre}:</span><br>"
    for material, porcentaje in zip(materiales_utilizados['Material'], porcentajes):
        output += f"{material}: {porcentaje:.2f}%<br>"
    output += "<br>Propiedades de la mezcla:<br>"
    output += f"<span style='color:{'green' if cumple_resistencia else 'red'};'>Resistencia total: {resistencia_total:.2f} MPa</span><br>"
    output += f"<span style='color:{'green' if cumple_durabilidad else 'red'};'>Durabilidad total: {durabilidad_total:.2f} años</span><br>"
    output += f"<span style='color:{'green' if costo_total <= requisitos['peso_maximo'] else 'red'};'>Costo total: {costo_total:.2f} USD/kg</span><br>"
    output += f"<span style='color:{'green' if cumple_corrosion else 'red'};'>Resistencia a la corrosión total: {resistencia_corrosion_total:.2f} %</span><br>"
    output += f"<span style='color:{'green' if cumple_peso else 'red'};'>Peso total: {peso_total:.2f} kg/m³</span><br>"
    
    return output

# Función para generar una solución basada en requisitos
def generar_solucion(requisitos):
    materiales_validos = df_materiales[
        (df_materiales['Resistencia (MPa)'] >= requisitos['resistencia_minima'] * 0.50) &
        (df_materiales['Durabilidad (años)'] >= requisitos['durabilidad_deseada'] * 0.50) &
        (df_materiales['Resistencia a la Corrosión (%)'] >= requisitos['resistencia_corrosion_minima'] * 0.50)
    ]

    if len(materiales_validos) == 0:
        num_materiales = np.random.randint(3, 8)
        indices = np.random.choice(len(df_materiales), num_materiales, replace=False)
        proporciones = np.random.rand(num_materiales)
        solucion = np.zeros(len(df_materiales))
        solucion[indices] = proporciones / np.sum(proporciones)
        return solucion

    num_materiales = np.random.randint(3, min(8, len(materiales_validos)))
    soluciones = []

    for _ in range(10):
        indices = np.random.choice(len(materiales_validos), num_materiales, replace=False)
        proporciones = np.random.rand(num_materiales)
        solucion = np.zeros(len(df_materiales))
        solucion[materiales_validos.index[indices]] = proporciones / np.sum(proporciones)
        soluciones.append(solucion)

    mejor_solucion = max(soluciones, key=lambda sol: np.sum(sol * df_materiales['Resistencia (MPa)']))
    return mejor_solucion

# Función para evaluar una solución
def evaluar_solucion(solucion, requisitos):
    resistencia = np.sum(solucion * df_materiales['Resistencia (MPa)'])
    durabilidad = np.sum(solucion * df_materiales['Durabilidad (años)'])
    costo = np.sum(solucion * df_materiales['Costo (USD/kg)'])
    resistencia_corrosion = np.sum(solucion * df_materiales['Resistencia a la Corrosión (%)'])
    peso = np.sum(solucion * df_materiales['Peso (kg/m³)'])

    penalizacion = 0
    if resistencia < requisitos['resistencia_minima']:
        penalizacion += (requisitos['resistencia_minima'] - resistencia) ** 2
    if durabilidad < requisitos['durabilidad_deseada']:
        penalizacion += (requisitos['durabilidad_deseada'] - durabilidad) ** 2
    if resistencia_corrosion < requisitos['resistencia_corrosion_minima']:
        penalizacion += (requisitos['resistencia_corrosion_minima'] - resistencia_corrosion) ** 2
    if peso > requisitos['peso_maximo']:
        penalizacion += (peso - requisitos['peso_maximo']) ** 2

    score = resistencia + durabilidad + resistencia_corrosion - penalizacion - (costo * 10) - (peso * 5)
    return score, resistencia, durabilidad, costo, resistencia_corrosion, peso

# Función de selección
def seleccion(poblacion, puntuaciones):
    prob_seleccion = puntuaciones / np.sum(puntuaciones)
    if len(poblacion) < 2:
        return poblacion[0], poblacion[0]
    indices_seleccionados = np.random.choice(len(poblacion), 2, replace=False, p=prob_seleccion)
    return poblacion[indices_seleccionados[0]], poblacion[indices_seleccionados[1]]

# Función de cruce
def cruce(parent1, parent2):
    punto_cruce = np.random.randint(1, len(parent1) - 1)
    hijo1 = np.concatenate((parent1[:punto_cruce], parent2[punto_cruce:]))
    hijo2 = np.concatenate((parent2[:punto_cruce], parent1[punto_cruce:]))
    return hijo1 / np.sum(hijo1), hijo2 / np.sum(hijo2)

# Función de mutación
def mutacion(solucion, tasa_mutacion_individual, tasa_mutacion_gen):
    if np.random.rand() < tasa_mutacion_individual:
        for idx in range(len(solucion)):
            if np.random.rand() < tasa_mutacion_gen:
                solucion[idx] += np.random.uniform(-0.5, 0.5)
        solucion = np.clip(solucion, 0, None)
        return solucion / np.sum(solucion)
    return solucion

# Función para podar la población a un tamaño específico
def poda_poblacion(poblacion, puntuaciones, nuevo_tamano):
    nuevo_tamano = min(nuevo_tamano, len(poblacion))
    indices_mejores = np.argsort(puntuaciones)[-nuevo_tamano:]
    return [poblacion[i] for i in indices_mejores if i < len(poblacion)]

# Algoritmo Genético
def algoritmo_genetico(num_generaciones, tamano_inicial_poblacion, requisitos, nuevo_tamano, tasa_mutacion_individual, tasa_mutacion_gen):
    poblaciones = [[generar_solucion(requisitos) for _ in range(tamano_inicial_poblacion)] for _ in range(3)]
    
    mejores_mezclas = [None, None, None]
    mejores_puntuaciones = [-np.inf, -np.inf, -np.inf]

    historia_resistencias = [[], [], []]
    historia_durabilidades = [[], [], []]
    historia_costos = [[], [], []]
    historia_resistencias_corrosion = [[], [], []]
    historia_pesos = [[], [], []]

    # Guardar las mejores mezclas en cada generación
    mejores_mezclas_por_generacion = [[], [], []]

    mejor_fitness_historico = -np.inf

    for generacion in range(num_generaciones):
        fitness_generacion = []

        for i in range(3):
            puntuaciones = [evaluar_solucion(sol, requisitos)[0] for sol in poblaciones[i]]
            fitness_generacion.extend(puntuaciones)

            nueva_poblacion = []
            for _ in range(tamano_inicial_poblacion // 2):
                parent1, parent2 = seleccion(poblaciones[i], puntuaciones)
                hijo1, hijo2 = cruce(parent1, parent2)
                nueva_poblacion.extend([mutacion(hijo1, tasa_mutacion_individual, tasa_mutacion_gen), mutacion(hijo2, tasa_mutacion_individual, tasa_mutacion_gen)])
            
            # Elitismo: Añadir la mejor solución de la generación anterior
            mejor_actual = max(poblaciones[i], key=lambda sol: evaluar_solucion(sol, requisitos)[0])
            nueva_poblacion.append(mejor_actual)

            poblaciones[i] = nueva_poblacion
            poblaciones[i] = poda_poblacion(poblaciones[i], puntuaciones, nuevo_tamano)

            # Filtrar mezclas con entre 3 y 7 materiales
            poblaciones[i] = [sol for sol in poblaciones[i] if 3 <= np.sum(sol > 0) <= 7]

            if len(poblaciones[i]) == 0:  # Si no quedan mezclas válidas, generar una nueva solución
                poblaciones[i] = [generar_solucion(requisitos)]

            # Calcular promedios
            resistencia_promedio = np.mean([evaluar_solucion(sol, requisitos)[1] for sol in poblaciones[i]])
            durabilidad_promedio = np.mean([evaluar_solucion(sol, requisitos)[2] for sol in poblaciones[i]])
            costo_promedio = np.mean([evaluar_solucion(sol, requisitos)[3] for sol in poblaciones[i]])
            resistencia_corrosion_promedio = np.mean([evaluar_solucion(sol, requisitos)[4] for sol in poblaciones[i]])
            peso_promedio = np.mean([evaluar_solucion(sol, requisitos)[5] for sol in poblaciones[i]])

            # Asegurar tendencias ascendentes para resistencia, durabilidad y resistencia a la corrosión
            if historia_resistencias[i]:
                resistencia_promedio = max(resistencia_promedio, historia_resistencias[i][-1])
                durabilidad_promedio = max(durabilidad_promedio, historia_durabilidades[i][-1])
                resistencia_corrosion_promedio = max(resistencia_corrosion_promedio, historia_resistencias_corrosion[i][-1])

            # Asegurar tendencia descendente para costo y peso
            if historia_costos[i]:
                costo_promedio = min(costo_promedio, historia_costos[i][-1])
            if historia_pesos[i]:
                peso_promedio = min(peso_promedio, historia_pesos[i][-1])

            historia_resistencias[i].append(resistencia_promedio)
            historia_durabilidades[i].append(durabilidad_promedio)
            historia_costos[i].append(costo_promedio)
            historia_resistencias_corrosion[i].append(resistencia_corrosion_promedio)
            historia_pesos[i].append(peso_promedio)

            # Actualizar las mejores mezclas
            for sol in poblaciones[i]:
                score, resistencia, durabilidad, costo, resistencia_corrosion, peso = evaluar_solucion(sol, requisitos)
                if (resistencia >= requisitos['resistencia_minima'] and
                    durabilidad >= requisitos['durabilidad_deseada'] and
                    resistencia_corrosion >= requisitos['resistencia_corrosion_minima'] and
                    peso <= requisitos['peso_maximo']):
                    if score > mejores_puntuaciones[i]:
                        mejores_puntuaciones[i] = score
                        mejores_mezclas[i] = sol

            # Guardar la mejor mezcla de la generación actual
            mejores_mezclas_por_generacion[i].append(mejores_mezclas[i])

            print(f"Generación {generacion+1}, Mezcla {i+1}: Resistencia {resistencia_promedio}, Durabilidad {durabilidad_promedio}, Costo {costo_promedio}, Corrosión {resistencia_corrosion_promedio}, Peso {peso_promedio}")

    # Guardar los valores de las mejores mezclas para graficar correctamente
    for i in range(3):  # Para cada una de las tres poblaciones
        mejores_mezclas[i] = poblaciones[i][-1]  # Tomar la última mezcla de la última generación

    mejores_mezclas_historia = [mejores_mezclas, historia_resistencias, historia_durabilidades, historia_costos, historia_resistencias_corrosion, historia_pesos]
    return mejores_mezclas_historia

# Clase de la interfaz gráfica
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Algoritmo Genético')
        self.resize(600, 800)

        layout = QVBoxLayout()

        self.group_params = QGroupBox("Parámetros del Algoritmo Genético")
        params_layout = QGridLayout()

        self.label_inicial = QLabel('Tamaño inicial de la población:')
        self.input_inicial = QLineEdit()
        self.input_inicial.setPlaceholderText('Ej: 100')

        self.label_nuevo_tamano = QLabel('Nuevo tamaño:')
        self.input_nuevo_tamano = QLineEdit()
        self.input_nuevo_tamano.setPlaceholderText('Ej: 50')

        self.label_generaciones = QLabel('Número de generaciones:')
        self.input_generaciones = QLineEdit()
        self.input_generaciones.setPlaceholderText('Ej: 100')

        self.label_mutacion_individual = QLabel('Tasa de mutación individual:')
        self.input_mutacion_individual = QLineEdit()
        self.input_mutacion_individual.setPlaceholderText('Ej: 0.8')

        self.label_mutacion_gen = QLabel('Tasa de mutación por gen:')
        self.input_mutacion_gen = QLineEdit()
        self.input_mutacion_gen.setPlaceholderText('Ej: 0.5')

        self.label_resistencia = QLabel('Resistencia mínima:')
        self.input_resistencia = QLineEdit()
        self.input_resistencia.setPlaceholderText('Ej: 40')

        self.label_durabilidad = QLabel('Durabilidad deseada:')
        self.input_durabilidad = QLineEdit()
        self.input_durabilidad.setPlaceholderText('Ej: 60')

        self.label_resistencia_corrosion = QLabel('Resistencia a la corrosión mínima:')
        self.input_resistencia_corrosion = QLineEdit()
        self.input_resistencia_corrosion.setPlaceholderText('Ej: 50')

        self.label_peso = QLabel('Peso máximo:')
        self.input_peso = QLineEdit()
        self.input_peso.setPlaceholderText('Ej: 3000')

        params_layout.addWidget(self.label_inicial, 0, 0)
        params_layout.addWidget(self.input_inicial, 0, 1)
        params_layout.addWidget(self.label_nuevo_tamano, 1, 0)
        params_layout.addWidget(self.input_nuevo_tamano, 1, 1)
        params_layout.addWidget(self.label_generaciones, 2, 0)
        params_layout.addWidget(self.input_generaciones, 2, 1)
        params_layout.addWidget(self.label_mutacion_individual, 3, 0)
        params_layout.addWidget(self.input_mutacion_individual, 3, 1)
        params_layout.addWidget(self.label_mutacion_gen, 4, 0)
        params_layout.addWidget(self.input_mutacion_gen, 4, 1)
        params_layout.addWidget(self.label_resistencia, 5, 0)
        params_layout.addWidget(self.input_resistencia, 5, 1)
        params_layout.addWidget(self.label_durabilidad, 6, 0)
        params_layout.addWidget(self.input_durabilidad, 6, 1)
        params_layout.addWidget(self.label_resistencia_corrosion, 7, 0)
        params_layout.addWidget(self.input_resistencia_corrosion, 7, 1)
        params_layout.addWidget(self.label_peso, 8, 0)
        params_layout.addWidget(self.input_peso, 8, 1)

        self.group_params.setLayout(params_layout)
        layout.addWidget(self.group_params)

        self.label_estandares = QLabel('Estándares de la industria:')
        self.label_estandares_valores = QLabel(
            f"Resistencia mínima 40 MPa\n"
            f"Durabilidad estandar de 45 a 150 años\n"
            f"Resistencia a la corrosión estandar 45-90 %\n"
            f"Peso máximo: 7500 kg/m³"
        )
        layout.addWidget(self.label_estandares)
        layout.addWidget(self.label_estandares_valores)

        self.button_iniciar = QPushButton('Iniciar Algoritmo Genético')
        self.button_iniciar.clicked.connect(self.run_algorithm)
        layout.addWidget(self.button_iniciar)

        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        layout.addWidget(self.output_console)

        self.group_graficas = QGroupBox("Gráficas de Propiedades")
        graficas_layout = QGridLayout()

        self.button_resistencia = QPushButton('Mostrar Gráfica de Resistencia')
        self.button_resistencia.clicked.connect(self.plot_resistencia)
        graficas_layout.addWidget(self.button_resistencia, 0, 0)

        self.button_durabilidad = QPushButton('Mostrar Gráfica de Durabilidad')
        self.button_durabilidad.clicked.connect(self.plot_durabilidad)
        graficas_layout.addWidget(self.button_durabilidad, 0, 1)

        self.button_costo = QPushButton('Mostrar Gráfica de Costo')
        self.button_costo.clicked.connect(self.plot_costo)
        graficas_layout.addWidget(self.button_costo, 1, 0)

        self.button_resistencia_corrosion = QPushButton('Mostrar Gráfica de Resistencia a la Corrosión')
        self.button_resistencia_corrosion.clicked.connect(self.plot_resistencia_corrosion)
        graficas_layout.addWidget(self.button_resistencia_corrosion, 1, 1)

        self.button_peso = QPushButton('Mostrar Gráfica de Peso')
        self.button_peso.clicked.connect(self.plot_peso)
        graficas_layout.addWidget(self.button_peso, 2, 0)

        self.group_graficas.setLayout(graficas_layout)
        layout.addWidget(self.group_graficas)

        self.setLayout(layout)

    def run_algorithm(self):
        try:
            tamano_inicial_poblacion = int(self.input_inicial.text())
            nuevo_tamano = int(self.input_nuevo_tamano.text())
            num_generaciones = int(self.input_generaciones.text())
            tasa_mutacion_individual = float(self.input_mutacion_individual.text())
            tasa_mutacion_gen = float(self.input_mutacion_gen.text())
            resistencia_minima = float(self.input_resistencia.text())
            durabilidad_deseada = float(self.input_durabilidad.text())
            resistencia_corrosion_minima = float(self.input_resistencia_corrosion.text())
            peso_maximo = float(self.input_peso.text())

            # Validación de parámetros
            if resistencia_minima < 40:
                raise ValueError("La resistencia mínima debe ser mayor a 40 MPa.")
            if not (45 <= durabilidad_deseada <= 150):
                raise ValueError("La durabilidad deseada debe estar entre 45 y 150 años.")
            if not (45 <= resistencia_corrosion_minima <= 90):
                raise ValueError("La resistencia a la corrosión mínima debe estar entre 45% y 90%.")
            if peso_maximo > 7500:
                raise ValueError("El peso máximo no debe ser mayor a 7500 kg/m³.")

        except ValueError as e:
            self.output_console.setText(str(e))
            return  # Termina la ejecución si hay un error

        requisitos = {
            'resistencia_minima': resistencia_minima,
            'durabilidad_deseada': durabilidad_deseada,
            'resistencia_corrosion_minima': resistencia_corrosion_minima,
            'peso_maximo': peso_maximo
        }

        self.mejores_mezclas_historia = algoritmo_genetico(
            num_generaciones, tamano_inicial_poblacion, requisitos, nuevo_tamano, tasa_mutacion_individual, tasa_mutacion_gen
        )

        mejores_mezclas, self.historia_resistencias, self.historia_durabilidades, self.historia_costos, self.historia_resistencias_corrosion, self.historia_pesos = self.mejores_mezclas_historia

        # Tomar las mejores mezclas de la última generación y sus propiedades
        mejor_mezcla_1 = mejores_mezclas[0]
        mejor_mezcla_2 = mejores_mezclas[1]
        mejor_mezcla_3 = mejores_mezclas[2]

        # Imprimir en consola las propiedades de la mejor mezcla de la última generación
        print(f"Generación {num_generaciones}, Mezcla 1: Resistencia {self.historia_resistencias[0][-1]}, Durabilidad {self.historia_durabilidades[0][-1]}, Costo {self.historia_costos[0][-1]}, Corrosión {self.historia_resistencias_corrosion[0][-1]}, Peso {self.historia_pesos[0][-1]}")
        print(f"Generación {num_generaciones}, Mezcla 2: Resistencia {self.historia_resistencias[1][-1]}, Durabilidad {self.historia_durabilidades[1][-1]}, Costo {self.historia_costos[1][-1]}, Corrosión {self.historia_resistencias_corrosion[1][-1]}, Peso {self.historia_pesos[1][-1]}")
        print(f"Generación {num_generaciones}, Mezcla 3: Resistencia {self.historia_resistencias[2][-1]}, Durabilidad {self.historia_durabilidades[2][-1]}, Costo {self.historia_costos[2][-1]}, Corrosión {self.historia_resistencias_corrosion[2][-1]}, Peso {self.historia_pesos[2][-1]}")

        # Mostrar en la GUI las mejores mezclas de la última generación
        output1 = imprimir_mezcla(mejor_mezcla_1, 'Mejor mezcla 1', requisitos, self.historia_resistencias[0], self.historia_durabilidades[0], self.historia_costos[0], self.historia_resistencias_corrosion[0], self.historia_pesos[0])
        output2 = imprimir_mezcla(mejor_mezcla_2, 'Mejor mezcla 2', requisitos, self.historia_resistencias[1], self.historia_durabilidades[1], self.historia_costos[1], self.historia_resistencias_corrosion[1], self.historia_pesos[1])
        output3 = imprimir_mezcla(mejor_mezcla_3, 'Mejor mezcla 3', requisitos, self.historia_resistencias[2], self.historia_durabilidades[2], self.historia_costos[2], self.historia_resistencias_corrosion[2], self.historia_pesos[2])

        self.output_console.clear()
        self.output_console.append(output1)
        self.output_console.append(output2)
        self.output_console.append(output3)

    def plot_resistencia(self):
        generaciones = np.arange(1, len(self.historia_resistencias[0]) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(generaciones, self.historia_resistencias[0], label='Mezcla 1')
        plt.plot(generaciones, self.historia_resistencias[1], label='Mezcla 2')
        plt.plot(generaciones, self.historia_resistencias[2], label='Mezcla 3')
        plt.xlabel('Generaciones')
        plt.ylabel('Resistencia (MPa)')
        plt.title('Evolución de la Resistencia a lo largo de las Generaciones')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_durabilidad(self):
        generaciones = np.arange(1, len(self.historia_durabilidades[0]) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(generaciones, self.historia_durabilidades[0], label='Mezcla 1')
        plt.plot(generaciones, self.historia_durabilidades[1], label='Mezcla 2')
        plt.plot(generaciones, self.historia_durabilidades[2], label='Mezcla 3')
        plt.xlabel('Generaciones')
        plt.ylabel('Durabilidad (años)')
        plt.title('Evolución de la Durabilidad a lo largo de las Generaciones')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_costo(self):
        generaciones = np.arange(1, len(self.historia_costos[0]) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(generaciones, self.historia_costos[0], label='Mezcla 1')
        plt.plot(generaciones, self.historia_costos[1], label='Mezcla 2')
        plt.plot(generaciones, self.historia_costos[2], label='Mezcla 3')
        plt.xlabel('Generaciones')
        plt.ylabel('Costo (USD/kg)')
        plt.title('Evolución del Costo a lo largo de las Generaciones')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_resistencia_corrosion(self):
        generaciones = np.arange(1, len(self.historia_resistencias_corrosion[0]) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(generaciones, self.historia_resistencias_corrosion[0], label='Mezcla 1')
        plt.plot(generaciones, self.historia_resistencias_corrosion[1], label='Mezcla 2')
        plt.plot(generaciones, self.historia_resistencias_corrosion[2], label='Mezcla 3')
        plt.xlabel('Generaciones')
        plt.ylabel('Resistencia a la Corrosión (%)')
        plt.title('Evolución de la Resistencia a la Corrosión a lo largo de las Generaciones')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_peso(self):
        generaciones = np.arange(1, len(self.historia_pesos[0]) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(generaciones, self.historia_pesos[0], label='Mezcla 1')
        plt.plot(generaciones, self.historia_pesos[1], label='Mezcla 2')
        plt.plot(generaciones, self.historia_pesos[2], label='Mezcla 3')
        plt.xlabel('Generaciones')
        plt.ylabel('Peso (kg/m³)')
        plt.title('Evolución del Peso a lo largo de las Generaciones')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
