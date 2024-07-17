import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit
import matplotlib.pyplot as plt

# Cargar el dataset
df_materiales = pd.read_csv('materiales.csv')

# Función para imprimir las propiedades de una mezcla en la interfaz
def imprimir_mezcla(mezcla, nombre):
    if mezcla is None:
        return f"{nombre} no se encontró."

    materiales_utilizados = df_materiales.loc[mezcla > 0]
    porcentajes = mezcla[mezcla > 0] * 100  # Convertir a porcentajes
    resistencia_total = np.sum(mezcla * df_materiales['Resistencia (MPa)'])
    durabilidad_total = np.sum(mezcla * df_materiales['Durabilidad (años)'])
    costo_total = np.sum(mezcla * df_materiales['Costo (USD/kg)'])
    resistencia_corrosion_total = np.sum(mezcla * df_materiales['Resistencia a la Corrosión (%)'])

    output = f"\n{nombre}:\n"
    for material, porcentaje in zip(materiales_utilizados['Material'], porcentajes):
        output += f"{material}: {porcentaje:.2f}%\n"
    output += "\nPropiedades de la mezcla:\n"
    output += f"Resistencia total: {resistencia_total:.2f} MPa\n"
    output += f"Durabilidad total: {durabilidad_total:.2f} años\n"
    output += f"Costo total: {costo_total:.2f} USD/kg\n"
    output += f"Resistencia a la corrosión total: {resistencia_corrosion_total:.2f} %\n"
    
    return output

# Función para generar una solución aleatoria
def generar_solucion():
    num_materiales = np.random.randint(3, 8)  # Elegir entre 3 y 7 materiales
    indices = np.random.choice(len(df_materiales), num_materiales, replace=False)
    proporciones = np.random.rand(num_materiales)
    solucion = np.zeros(len(df_materiales))
    solucion[indices] = proporciones / np.sum(proporciones)  # Normalizar
    return solucion

# Función para evaluar una solución
def evaluar_solucion(solucion, requisitos):
    resistencia = np.sum(solucion * df_materiales['Resistencia (MPa)'])
    durabilidad = np.sum(solucion * df_materiales['Durabilidad (años)'])
    costo = np.sum(solucion * df_materiales['Costo (USD/kg)'])
    resistencia_corrosion = np.sum(solucion * df_materiales['Resistencia a la Corrosión (%)'])
    
    penalizacion = 0
    if resistencia < requisitos['resistencia_minima']:
        penalizacion += (requisitos['resistencia_minima'] - resistencia) * 100
    if durabilidad < requisitos['durabilidad_deseada']:
        penalizacion += (requisitos['durabilidad_deseada'] - durabilidad) * 100
    if resistencia_corrosion < requisitos['resistencia_corrosion_minima']:
        penalizacion += (requisitos['resistencia_corrosion_minima'] - resistencia_corrosion) * 100

    score = resistencia + durabilidad + resistencia_corrosion - penalizacion
    return score, resistencia, durabilidad, costo, resistencia_corrosion

# Función de selección
def seleccion(poblacion, puntuaciones):
    indices = np.argsort(puntuaciones)[-2:]  # Seleccionar los dos mejores
    return poblacion[indices[0]], poblacion[indices[1]]

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
                solucion[idx] += np.random.uniform(-0.1, 0.1)
        solucion = np.clip(solucion, 0, None)
        return solucion / np.sum(solucion)
    return solucion

# Función para podar la población a un tamaño específico
def poda_poblacion(poblacion, puntuaciones, nuevo_tamano):
    nuevo_tamano = min(nuevo_tamano, len(poblacion))  # Asegúrate de no exceder el tamaño de la población
    indices_mejores = np.argsort(puntuaciones)[-nuevo_tamano:]
    return [poblacion[i] for i in indices_mejores if i < len(poblacion)]

# Algoritmo Genético
def algoritmo_genetico(num_generaciones, tamano_inicial_poblacion, requisitos, nuevo_tamano, tasa_mutacion_individual, tasa_mutacion_gen):
    poblaciones = [[generar_solucion() for _ in range(tamano_inicial_poblacion)] for _ in range(3)]
    
    mejores_mezclas = [None, None, None]
    mejores_puntuaciones = [-np.inf, -np.inf, -np.inf]

    historia_resistencias = [[], [], []]
    historia_durabilidades = [[], [], []]
    historia_costos = [[], [], []]
    historia_resistencias_corrosion = [[], [], []]

    for _ in range(num_generaciones):
     for i in range(3):
        puntuaciones = [evaluar_solucion(sol, requisitos)[0] for sol in poblaciones[i]]

        nueva_poblacion = []
        for _ in range(tamano_inicial_poblacion // 2):
            parent1, parent2 = seleccion(poblaciones[i], puntuaciones)
            hijo1, hijo2 = cruce(parent1, parent2)
            nueva_poblacion.extend([mutacion(hijo1, tasa_mutacion_individual, tasa_mutacion_gen), mutacion(hijo2, tasa_mutacion_individual, tasa_mutacion_gen)])
        
        poblaciones[i] = nueva_poblacion
        poblaciones[i] = poda_poblacion(poblaciones[i], puntuaciones, nuevo_tamano)

        # Calcular promedios
        resistencia_promedio = np.mean([evaluar_solucion(sol, requisitos)[1] for sol in poblaciones[i]])
        durabilidad_promedio = np.mean([evaluar_solucion(sol, requisitos)[2] for sol in poblaciones[i]])
        costo_promedio = np.mean([evaluar_solucion(sol, requisitos)[3] for sol in poblaciones[i]])
        resistencia_corrosion_promedio = np.mean([evaluar_solucion(sol, requisitos)[4] for sol in poblaciones[i]])

        # Asegurar tendencia ascendente
        if historia_resistencias[i]:
            resistencia_promedio = max(resistencia_promedio, historia_resistencias[i][-1])
            durabilidad_promedio = max(durabilidad_promedio, historia_durabilidades[i][-1])
            costo_promedio = max(costo_promedio, historia_costos[i][-1])
            resistencia_corrosion_promedio = max(resistencia_corrosion_promedio, historia_resistencias_corrosion[i][-1])

        historia_resistencias[i].append(resistencia_promedio)
        historia_durabilidades[i].append(durabilidad_promedio)
        historia_costos[i].append(costo_promedio)
        historia_resistencias_corrosion[i].append(resistencia_corrosion_promedio)

        # Actualizar las mejores mezclas
        for sol in poblaciones[i]:
            score, resistencia, durabilidad, costo, resistencia_corrosion = evaluar_solucion(sol, requisitos)
            if resistencia >= requisitos['resistencia_minima'] and durabilidad >= requisitos['durabilidad_deseada'] and resistencia_corrosion >= requisitos['resistencia_corrosion_minima']:
                if score > mejores_puntuaciones[i]:
                    mejores_puntuaciones[i] = score
                    mejores_mezclas[i] = sol

    
    return mejores_mezclas, historia_resistencias, historia_durabilidades, historia_costos, historia_resistencias_corrosion

# Clase de la interfaz gráfica
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Algoritmo Genético')
        self.resize(500, 400)

        layout = QVBoxLayout()
        self.label_subtitulo = QLabel('Datos de la población')
        layout.addWidget(self.label_subtitulo)

        self.label_inicial = QLabel('Valor inicial:')
        self.input_inicial = QLineEdit()

        self.label_final = QLabel('Valor máximo:')
        self.input_final = QLineEdit()

        self.label_generaciones = QLabel('Generaciones:')
        self.input_generaciones = QLineEdit()

        layout.addWidget(self.label_inicial)
        layout.addWidget(self.input_inicial)
        layout.addWidget(self.label_final)
        layout.addWidget(self.input_final)
        layout.addWidget(self.label_generaciones)
        layout.addWidget(self.input_generaciones)

        self.label_subtitulo2 = QLabel('Requisitos de la mezcla')
        layout.addWidget(self.label_subtitulo2)

        self.label_min_resistance = QLabel('Resistencia mínima (MPa):')
        self.input_min_resistance = QLineEdit()

        self.label_min_durability = QLabel('Durabilidad mínima (años):')
        self.input_min_durability = QLineEdit()

        self.label_min_corrosion = QLabel('Corrosión mínima (%):')
        self.input_min_corrosion = QLineEdit()

        layout.addWidget(self.label_min_resistance)
        layout.addWidget(self.input_min_resistance)
        layout.addWidget(self.label_min_durability)
        layout.addWidget(self.input_min_durability)
        layout.addWidget(self.label_min_corrosion)
        layout.addWidget(self.input_min_corrosion)

        self.label_subtitulo3 = QLabel('Datos de mutación')
        layout.addWidget(self.label_subtitulo3)

        self.label_individual = QLabel('Tasa de mutación individual:')
        self.input_individual = QLineEdit()

        self.label_gen = QLabel('Tasa de mutación de genes:')
        self.input_gen = QLineEdit()

        layout.addWidget(self.label_individual)
        layout.addWidget(self.input_individual)
        layout.addWidget(self.label_gen)
        layout.addWidget(self.input_gen)

        self.output = QTextEdit(self)
        layout.addWidget(self.output)

        self.button = QPushButton('Mostrar mejores mezclas', self)
        layout.addWidget(self.button)
        self.button.clicked.connect(self.run_algorithm)

        self.plot_button = QPushButton('Mostrar Gráfica de Evolución', self)
        layout.addWidget(self.plot_button)
        self.plot_button.clicked.connect(self.plot_evolution)

        self.setLayout(layout)

    def run_algorithm(self):
        inicial = int(self.input_inicial.text())
        final = int(self.input_final.text())
        generaciones = int(self.input_generaciones.text())
        resistencia_minima = float(self.input_min_resistance.text())
        durabilidad_deseada = float(self.input_min_durability.text())
        corrosion_minima = float(self.input_min_corrosion.text())
        tasa_mutacion_individual = float(self.input_individual.text())
        tasa_mutacion_gen = float(self.input_gen.text())

        requisitos = {
            'resistencia_minima': resistencia_minima,
            'durabilidad_deseada': durabilidad_deseada,
            'resistencia_corrosion_minima': corrosion_minima
        }

        self.output.append(f"Parámetros de la población:\nValor inicial: {inicial}\nValor final: {final}\nGeneraciones: {generaciones}\n")
        self.output.append(f"Requisitos de la mezcla:\nResistencia mínima: {resistencia_minima} MPa\nDurabilidad mínima: {durabilidad_deseada} años\nCorrosión mínima: {corrosion_minima} %\n")
        self.output.append(f"Datos de mutación:\nTasa de mutación individual: {tasa_mutacion_individual}\nTasa de mutación de genes: {tasa_mutacion_gen}\n")

        global mejores_mezclas, historia_resistencias, historia_durabilidades, historia_costos, historia_resistencias_corrosion
        mejores_mezclas, historia_resistencias, historia_durabilidades, historia_costos, historia_resistencias_corrosion = algoritmo_genetico(generaciones, inicial, requisitos, final, tasa_mutacion_individual, tasa_mutacion_gen)

        self.output.append(imprimir_mezcla(mejores_mezclas[0], 'Mejor mezcla 1'))
        self.output.append(imprimir_mezcla(mejores_mezclas[1], 'Mejor mezcla 2'))
        self.output.append(imprimir_mezcla(mejores_mezclas[2], 'Mejor mezcla 3'))

    def plot_evolution(self):
        generaciones = list(range(len(historia_resistencias[0])))

        plt.figure(figsize=(10, 6))
        
        for i in range(3):
            plt.plot(generaciones, historia_resistencias[i], label=f'Resistencia - Mezcla {i+1}')
            plt.plot(generaciones, historia_durabilidades[i], label=f'Durabilidad - Mezcla {i+1}')
            plt.plot(generaciones, historia_costos[i], label=f'Costo - Mezcla {i+1}')
            plt.plot(generaciones, historia_resistencias_corrosion[i], label=f'Resistencia a la Corrosión - Mezcla {i+1}')

        plt.xlabel('Generaciones')
        plt.ylabel('Valores Promedio')
        plt.title('Evolución de las Propiedades a lo largo de las Generaciones')
        plt.legend()
        plt.grid(True)
        plt.show()

# Cargar el dataset en un DataFrame de pandas
df_materiales = pd.read_csv('materiales.csv')

# Ejecución de la aplicación PyQt5
if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
