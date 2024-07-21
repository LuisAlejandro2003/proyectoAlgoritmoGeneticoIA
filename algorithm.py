import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit
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
    output += f"Resistencia total: {resistencia_total:.2f} MPa<br>"
    output += f"Durabilidad total: {durabilidad_total:.2f} años<br>"
    output += f"Costo total: {costo_total:.2f} USD/kg<br>"
    output += f"Resistencia a la corrosión total: {resistencia_corrosion_total:.2f} %<br>"
    output += f"Peso total: {peso_total:.2f} kg/m³<br>"
    
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

    for generacion in range(num_generaciones):
        for i in range(3):
            puntuaciones = [evaluar_solucion(sol, requisitos)[0] for sol in poblaciones[i]]

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
        self.resize(500, 400)

        layout = QVBoxLayout()
        self.label_subtitulo = QLabel('Parámetros del algoritmo genético')
        layout.addWidget(self.label_subtitulo)

        self.label_inicial = QLabel('Tamaño inicial de la población:')
        self.input_inicial = QLineEdit()
        self.input_inicial.setPlaceholderText('Tamaño inicial de la población')
        layout.addWidget(self.label_inicial)
        layout.addWidget(self.input_inicial)

        self.label_nuevo_tamano = QLabel('Nuevo tamaño:')
        self.input_nuevo_tamano = QLineEdit()
        self.input_nuevo_tamano.setPlaceholderText('Nuevo tamaño de la población')
        layout.addWidget(self.label_nuevo_tamano)
        layout.addWidget(self.input_nuevo_tamano)

        self.label_generaciones = QLabel('Número de generaciones:')
        self.input_generaciones = QLineEdit()
        self.input_generaciones.setPlaceholderText('Número de generaciones')
        layout.addWidget(self.label_generaciones)
        layout.addWidget(self.input_generaciones)

        self.label_mutacion_individual = QLabel('Tasa de mutación individual:')
        self.input_mutacion_individual = QLineEdit()
        self.input_mutacion_individual.setPlaceholderText('Tasa de mutación individual')
        layout.addWidget(self.label_mutacion_individual)
        layout.addWidget(self.input_mutacion_individual)

        self.label_mutacion_gen = QLabel('Tasa de mutación por gen:')
        self.input_mutacion_gen = QLineEdit()
        self.input_mutacion_gen.setPlaceholderText('Tasa de mutación por gen')
        layout.addWidget(self.label_mutacion_gen)
        layout.addWidget(self.input_mutacion_gen)

        self.label_resistencia = QLabel('Resistencia mínima:')
        self.input_resistencia = QLineEdit()
        self.input_resistencia.setPlaceholderText('Resistencia mínima')
        layout.addWidget(self.label_resistencia)
        layout.addWidget(self.input_resistencia)

        self.label_durabilidad = QLabel('Durabilidad deseada:')
        self.input_durabilidad = QLineEdit()
        self.input_durabilidad.setPlaceholderText('Durabilidad deseada')
        layout.addWidget(self.label_durabilidad)
        layout.addWidget(self.input_durabilidad)

        self.label_resistencia_corrosion = QLabel('Resistencia a la corrosión mínima:')
        self.input_resistencia_corrosion = QLineEdit()
        self.input_resistencia_corrosion.setPlaceholderText('Resistencia a la corrosión mínima')
        layout.addWidget(self.label_resistencia_corrosion)
        layout.addWidget(self.input_resistencia_corrosion)

        self.label_peso = QLabel('Peso máximo:')
        self.input_peso = QLineEdit()
        self.input_peso.setPlaceholderText('Peso máximo')
        layout.addWidget(self.label_peso)
        layout.addWidget(self.input_peso)

        self.button_iniciar = QPushButton('Iniciar Algoritmo Genético')
        self.button_iniciar.clicked.connect(self.run_algorithm)
        layout.addWidget(self.button_iniciar)

        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        layout.addWidget(self.output_console)

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
        except ValueError:
            self.output_console.setText("Por favor, asegúrate de que todos los campos están llenos y contienen valores numéricos válidos.")
            return  # Termina la ejecución si hay un error

        requisitos = {
            'resistencia_minima': resistencia_minima,
            'durabilidad_deseada': durabilidad_deseada,
            'resistencia_corrosion_minima': resistencia_corrosion_minima,
            'peso_maximo': peso_maximo
        }

        mejores_mezclas_historia = algoritmo_genetico(
            num_generaciones, tamano_inicial_poblacion, requisitos, nuevo_tamano, tasa_mutacion_individual, tasa_mutacion_gen
        )

        mejores_mezclas, historia_resistencias, historia_durabilidades, historia_costos, historia_resistencias_corrosion, historia_pesos = mejores_mezclas_historia

        # Tomar las mejores mezclas de la última generación y sus propiedades
        mejor_mezcla_1 = mejores_mezclas[0]
        mejor_mezcla_2 = mejores_mezclas[1]
        mejor_mezcla_3 = mejores_mezclas[2]

        # Imprimir en consola las propiedades de la mejor mezcla de la última generación
        print(f"Generación {num_generaciones}, Mezcla 1: Resistencia {historia_resistencias[0][-1]}, Durabilidad {historia_durabilidades[0][-1]}, Costo {historia_costos[0][-1]}, Corrosión {historia_resistencias_corrosion[0][-1]}, Peso {historia_pesos[0][-1]}")
        print(f"Generación {num_generaciones}, Mezcla 2: Resistencia {historia_resistencias[1][-1]}, Durabilidad {historia_durabilidades[1][-1]}, Costo {historia_costos[1][-1]}, Corrosión {historia_resistencias_corrosion[1][-1]}, Peso {historia_pesos[1][-1]}")
        print(f"Generación {num_generaciones}, Mezcla 3: Resistencia {historia_resistencias[2][-1]}, Durabilidad {historia_durabilidades[2][-1]}, Costo {historia_costos[2][-1]}, Corrosión {historia_resistencias_corrosion[2][-1]}, Peso {historia_pesos[2][-1]}")

        # Mostrar en la GUI las mejores mezclas de la última generación
        output1 = imprimir_mezcla(mejor_mezcla_1, 'Mejor mezcla 1', requisitos, historia_resistencias[0], historia_durabilidades[0], historia_costos[0], historia_resistencias_corrosion[0], historia_pesos[0])
        output2 = imprimir_mezcla(mejor_mezcla_2, 'Mejor mezcla 2', requisitos, historia_resistencias[1], historia_durabilidades[1], historia_costos[1], historia_resistencias_corrosion[1], historia_pesos[1])
        output3 = imprimir_mezcla(mejor_mezcla_3, 'Mejor mezcla 3', requisitos, historia_resistencias[2], historia_durabilidades[2], historia_costos[2], historia_resistencias_corrosion[2], historia_pesos[2])

        self.output_console.clear()
        self.output_console.append(output1)
        self.output_console.append(output2)
        self.output_console.append(output3)

        # Graficar la evolución de las propiedades
        plt.figure(figsize=(10, 8))
        generaciones = np.arange(1, num_generaciones + 1)
        plt.subplot(5, 1, 1)
        plt.plot(generaciones, historia_resistencias[0], label='Mezcla 1')
        plt.plot(generaciones, historia_resistencias[1], label='Mezcla 2')
        plt.plot(generaciones, historia_resistencias[2], label='Mezcla 3')
        plt.ylabel('Resistencia (MPa)')
        plt.legend()

        plt.subplot(5, 1, 2)
        plt.plot(generaciones, historia_durabilidades[0], label='Mezcla 1')
        plt.plot(generaciones, historia_durabilidades[1], label='Mezcla 2')
        plt.plot(generaciones, historia_durabilidades[2], label='Mezcla 3')
        plt.ylabel('Durabilidad (años)')
        plt.legend()

        plt.subplot(5, 1, 3)
        plt.plot(generaciones, historia_costos[0], label='Mezcla 1')
        plt.plot(generaciones, historia_costos[1], label='Mezcla 2')
        plt.plot(generaciones, historia_costos[2], label='Mezcla 3')
        plt.ylabel('Costo (USD/kg)')
        plt.legend()

        plt.subplot(5, 1, 4)
        plt.plot(generaciones, historia_resistencias_corrosion[0], label='Mezcla 1')
        plt.plot(generaciones, historia_resistencias_corrosion[1], label='Mezcla 2')
        plt.plot(generaciones, historia_resistencias_corrosion[2], label='Mezcla 3')
        plt.ylabel('Resistencia a la Corrosión (%)')
        plt.legend()

        plt.subplot(5, 1, 5)
        plt.plot(generaciones, historia_pesos[0], label='Mezcla 1')
        plt.plot(generaciones, historia_pesos[1], label='Mezcla 2')
        plt.plot(generaciones, historia_pesos[2], label='Mezcla 3')
        plt.xlabel('Generaciones')
        plt.ylabel('Peso (kg/m³)')
        plt.legend()

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
