import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextBrowser

# Cargar el dataset
df_materiales = pd.read_csv('materiales.csv')

# Función para imprimir las propiedades de una mezcla en la interfaz
def imprimir_mezcla(mezcla, nombre, requisitos):
    if mezcla is None:
        return f"<span style='color:red;'>{nombre} no cumple con los requisitos solicitados.</span>"

    materiales_utilizados = df_materiales.loc[mezcla > 0]
    porcentajes = mezcla[mezcla > 0] * 100  # Convertir a porcentajes
    resistencia_total = np.sum(mezcla * df_materiales['Resistencia (MPa)'])
    durabilidad_total = np.sum(mezcla * df_materiales['Durabilidad (años)'])
    costo_total = np.sum(mezcla * df_materiales['Costo (USD/kg)'])
    resistencia_corrosion_total = np.sum(mezcla * df_materiales['Resistencia a la Corrosión (%)'])
    peso_total = np.sum(mezcla * df_materiales['Peso (kg/m³)'])

    # Comprobar si cumple con los requisitos
    if (resistencia_total >= requisitos['resistencia_minima'] and
        durabilidad_total >= requisitos['durabilidad_deseada'] and
        resistencia_corrosion_total >= requisitos['resistencia_corrosion_minima'] and
        peso_total <= requisitos['peso_maximo']):
        color = 'green'
    else:
        color = 'red'

    output = f"<span style='color:{color};'>{nombre}:</span><br>"
    for material, porcentaje in zip(materiales_utilizados['Material'], porcentajes):
        output += f"{material}: {porcentaje:.2f}%<br>"
    output += "<br>Propiedades de la mezcla:<br>"
    output += f"<span style='color:{color};'>Resistencia total: {resistencia_total:.2f} MPa</span><br>"
    output += f"<span style='color:{color};'>Durabilidad total: {durabilidad_total:.2f} años</span><br>"
    output += f"<span style='color:{color};'>Costo total: {costo_total:.2f} USD/kg</span><br>"
    output += f"<span style='color:{color};'>Resistencia a la corrosión total: {resistencia_corrosion_total:.2f} %</span><br>"
    output += f"<span style='color:{color};'>Peso total: {peso_total:.2f} kg/m³</span><br>"
    
    return output

def generar_solucion(requisitos):
    # Seleccionar materiales que cumplen los requisitos en un rango
    materiales_validos = df_materiales[
        (df_materiales['Resistencia (MPa)'] >= requisitos['resistencia_minima'] * 0.75) &
        (df_materiales['Durabilidad (años)'] >= requisitos['durabilidad_deseada'] * 0.75) &
        (df_materiales['Resistencia a la Corrosión (%)'] >= requisitos['resistencia_corrosion_minima'] * 0.75) &
        (df_materiales['Peso (kg/m³)'] <= requisitos['peso_maximo'])
    ]

    if len(materiales_validos) == 0:
        # Generar solución aleatoria si no hay materiales válidos
        num_materiales = np.random.randint(3, 8)
        indices = np.random.choice(len(df_materiales), num_materiales, replace=False)
        proporciones = np.random.rand(num_materiales)
        solucion = np.zeros(len(df_materiales))
        solucion[indices] = proporciones / np.sum(proporciones)  # Normalizar
        return solucion

    # Generar soluciones iniciales
    num_materiales = np.random.randint(3, min(8, len(materiales_validos)))
    soluciones = []

    for _ in range(10):  # Generar 10 soluciones diversificadas
        indices = np.random.choice(len(materiales_validos), num_materiales, replace=False)
        proporciones = np.random.rand(num_materiales)
        solucion = np.zeros(len(df_materiales))
        solucion[materiales_validos.index[indices]] = proporciones / np.sum(proporciones)  # Normalizar
        soluciones.append(solucion)

    # Seleccionar la mejor solución inicial
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
        penalizacion += (requisitos['resistencia_minima'] - resistencia) * 100
    if durabilidad < requisitos['durabilidad_deseada']:
        penalizacion += (requisitos['durabilidad_deseada'] - durabilidad) * 100
    if resistencia_corrosion < requisitos['resistencia_corrosion_minima']:
        penalizacion += (requisitos['resistencia_corrosion_minima'] - resistencia_corrosion) * 100
    if peso > requisitos['peso_maximo']:
        penalizacion += (peso - requisitos['peso_maximo']) * 100

    score = resistencia + durabilidad + resistencia_corrosion - penalizacion
    return score, resistencia, durabilidad, costo, resistencia_corrosion, peso

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
                solucion[idx] += np.random.uniform(-0.5, 0.5)
        solucion = np.clip(solucion, 0, None)
        return solucion / np.sum(solucion)
    return solucion

# Función para podar la población a un tamaño específico
def poda_poblacion(poblacion, puntuaciones, nuevo_tamano):
    nuevo_tamano = min(nuevo_tamano, len(poblacion))  # Asegúrate de no exceder el tamaño de la población
    indices_mejores = np.argsort(puntuaciones)[-nuevo_tamano:]
    return [poblacion[i] for i in indices_mejores if i < len(poblacion)]

def generar_graficas(historia_resistencias, historia_durabilidades, historia_resistencias_corrosion, historia_pesos):
    generaciones = range(len(historia_resistencias[0]))

    for i in range(3):
        plt.figure(figsize=(10, 6))

        # Gráfica de Resistencia
        plt.subplot(2, 2, 1)
        plt.plot(generaciones, historia_resistencias[i], label='Resistencia')
        plt.xlabel('Generaciones')
        plt.ylabel('Resistencia (MPa)')
        plt.title(f'Historia de Resistencia - Mezcla {i+1}')
        plt.legend()

        # Gráfica de Durabilidad
        plt.subplot(2, 2, 2)
        plt.plot(generaciones, historia_durabilidades[i], label='Durabilidad', color='orange')
        plt.xlabel('Generaciones')
        plt.ylabel('Durabilidad (años)')
        plt.title(f'Historia de Durabilidad - Mezcla {i+1}')
        plt.legend()

        # Gráfica de Resistencia a la Corrosión
        plt.subplot(2, 2, 3)
        plt.plot(generaciones, historia_resistencias_corrosion[i], label='Resistencia a la Corrosión', color='green')
        plt.xlabel('Generaciones')
        plt.ylabel('Resistencia a la Corrosión (%)')
        plt.title(f'Historia de Resistencia a la Corrosión - Mezcla {i+1}')
        plt.legend()

        # Gráfica de Peso
        plt.subplot(2, 2, 4)
        plt.plot(generaciones, historia_pesos[i], label='Peso', color='red')
        plt.xlabel('Generaciones')
        plt.ylabel('Peso (kg/m³)')
        plt.title(f'Historia de Peso - Mezcla {i+1}')
        plt.legend()

        plt.tight_layout()
        plt.show()

# Modifica tu función algoritmo_genetico para devolver las historias de las mezclas
def algoritmo_genetico(num_generaciones, tamano_inicial_poblacion, requisitos, nuevo_tamano, tasa_mutacion_individual, tasa_mutacion_gen):
    poblaciones = [[generar_solucion(requisitos) for _ in range(tamano_inicial_poblacion)] for _ in range(3)]
    
    mejores_mezclas = [None, None, None]
    mejores_puntuaciones = [-np.inf, -np.inf, -np.inf]

    historia_resistencias = [[], [], []]
    historia_durabilidades = [[], [], []]
    historia_costos = [[], [], []]
    historia_resistencias_corrosion = [[], [], []]
    historia_pesos = [[], [], []]

    for _ in range(num_generaciones):
        for i in range(3):
            puntuaciones = [evaluar_solucion(sol, requisitos)[0] for sol in poblaciones[i]]

            nueva_poblacion = []
            for _ in range(tamano_inicial_poblacion // 2):
                parent1, parent2 = seleccion(poblaciones[i], puntuaciones)
                hijo1, hijo2 = cruce(parent1, parent2)
                nueva_poblacion.append(mutacion(hijo1, tasa_mutacion_individual, tasa_mutacion_gen))
                nueva_poblacion.append(mutacion(hijo2, tasa_mutacion_individual, tasa_mutacion_gen))

            poblaciones[i] = poda_poblacion(nueva_poblacion, puntuaciones, nuevo_tamano)

            # Guardar el mejor resultado
            puntuaciones = [evaluar_solucion(sol, requisitos)[0] for sol in poblaciones[i]]
            mejor_indice = np.argmax(puntuaciones)
            if puntuaciones[mejor_indice] > mejores_puntuaciones[i]:
                mejores_mezclas[i] = poblaciones[i][mejor_indice]
                mejores_puntuaciones[i] = puntuaciones[mejor_indice]

            # Guardar la historia de los resultados
            historia_resistencias[i].append(np.max([evaluar_solucion(sol, requisitos)[1] for sol in poblaciones[i]]))
            historia_durabilidades[i].append(np.max([evaluar_solucion(sol, requisitos)[2] for sol in poblaciones[i]]))
            historia_costos[i].append(np.min([evaluar_solucion(sol, requisitos)[3] for sol in poblaciones[i]]))
            historia_resistencias_corrosion[i].append(np.max([evaluar_solucion(sol, requisitos)[4] for sol in poblaciones[i]]))
            historia_pesos[i].append(np.min([evaluar_solucion(sol, requisitos)[5] for sol in poblaciones[i]]))

    # Generar gráficos al final
    generar_graficas(historia_resistencias, historia_durabilidades, historia_resistencias_corrosion, historia_pesos)

    return mejores_mezclas, historia_resistencias, historia_durabilidades, historia_costos, historia_resistencias_corrosion, historia_pesos

# Interfaz de usuario con PyQt5
class GeneticAlgorithmUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Algoritmo Genético - Optimización de Mezclas')
        self.layout = QVBoxLayout()
        
        self.resistencia_label = QLabel('Resistencia Mínima (MPa):')
        self.resistencia_input = QLineEdit()
        self.durabilidad_label = QLabel('Durabilidad Deseada (años):')
        self.durabilidad_input = QLineEdit()
        self.resistencia_corrosion_label = QLabel('Resistencia a la Corrosión Mínima (%):')
        self.resistencia_corrosion_input = QLineEdit()
        self.peso_maximo_label = QLabel('Peso Máximo (kg/m³):')
        self.peso_maximo_input = QLineEdit()
        self.generaciones_label = QLabel('Número de Generaciones:')
        self.generaciones_input = QLineEdit()
        self.tamano_inicial_label = QLabel('Tamaño Inicial de la Población:')
        self.tamano_inicial_input = QLineEdit()
        self.nuevo_tamano_label = QLabel('Nuevo Tamaño de la Población:')
        self.nuevo_tamano_input = QLineEdit()
        self.tasa_mutacion_individual_label = QLabel('Tasa de Mutación Individual:')
        self.tasa_mutacion_individual_input = QLineEdit()
        self.tasa_mutacion_gen_label = QLabel('Tasa de Mutación por Gen:')
        self.tasa_mutacion_gen_input = QLineEdit()
        
        self.resultado_text = QTextBrowser()
        self.ejecutar_button = QPushButton('Ejecutar Algoritmo')
        self.ejecutar_button.clicked.connect(self.ejecutar_algoritmo)
        
        self.layout.addWidget(self.resistencia_label)
        self.layout.addWidget(self.resistencia_input)
        self.layout.addWidget(self.durabilidad_label)
        self.layout.addWidget(self.durabilidad_input)
        self.layout.addWidget(self.resistencia_corrosion_label)
        self.layout.addWidget(self.resistencia_corrosion_input)
        self.layout.addWidget(self.peso_maximo_label)
        self.layout.addWidget(self.peso_maximo_input)
        self.layout.addWidget(self.generaciones_label)
        self.layout.addWidget(self.generaciones_input)
        self.layout.addWidget(self.tamano_inicial_label)
        self.layout.addWidget(self.tamano_inicial_input)
        self.layout.addWidget(self.nuevo_tamano_label)
        self.layout.addWidget(self.nuevo_tamano_input)
        self.layout.addWidget(self.tasa_mutacion_individual_label)
        self.layout.addWidget(self.tasa_mutacion_individual_input)
        self.layout.addWidget(self.tasa_mutacion_gen_label)
        self.layout.addWidget(self.tasa_mutacion_gen_input)
        self.layout.addWidget(self.ejecutar_button)
        self.layout.addWidget(self.resultado_text)
        
        self.setLayout(self.layout)
    
    def ejecutar_algoritmo(self):
        requisitos = {
            'resistencia_minima': float(self.resistencia_input.text()),
            'durabilidad_deseada': float(self.durabilidad_input.text()),
            'resistencia_corrosion_minima': float(self.resistencia_corrosion_input.text()),
            'peso_maximo': float(self.peso_maximo_input.text())
        }
        num_generaciones = int(self.generaciones_input.text())
        tamano_inicial_poblacion = int(self.tamano_inicial_input.text())
        nuevo_tamano = int(self.nuevo_tamano_input.text())
        tasa_mutacion_individual = float(self.tasa_mutacion_individual_input.text())
        tasa_mutacion_gen = float(self.tasa_mutacion_gen_input.text())
        
        mejores_mezclas, historia_resistencias, historia_durabilidades, historia_costos, historia_resistencias_corrosion, historia_pesos = algoritmo_genetico(
            num_generaciones,
            tamano_inicial_poblacion,
            requisitos,
            nuevo_tamano,
            tasa_mutacion_individual,
            tasa_mutacion_gen
        )
        
        resultados = ""
        for i, mezcla in enumerate(mejores_mezclas):
            resultados += f"<h2>Mejor mezcla {i+1}</h2>"
            resultados += imprimir_mezcla(mezcla, f"Mezcla {i+1}", requisitos)
        
        self.resultado_text.setHtml(resultados)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ventana = GeneticAlgorithmUI()
    ventana.show()
    sys.exit(app.exec_())
