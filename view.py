from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit
from algorithm import GeneticAlgorithm

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Algoritmo Genético')
        self.resize(500, 400)  # Aumentar el tamaño para incluir los nuevos campos

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

        self.label_subtitulo_mutacion = QLabel('Datos de mutación')
        self.label_subtitulo_mutacion.setStyleSheet('font-weight: bold')
        layout.addWidget(self.label_subtitulo_mutacion)

        self.label_mutacion_individual = QLabel('Probabilidad de mutación individual:')
        self.input_mutacion_individual = QLineEdit()
        layout.addWidget(self.label_mutacion_individual)
        layout.addWidget(self.input_mutacion_individual)

        self.label_mutacion_por_gen = QLabel('Probabilidad de mutación por gen:')
        self.input_mutacion_por_gen = QLineEdit()
        layout.addWidget(self.label_mutacion_por_gen)
        layout.addWidget(self.input_mutacion_por_gen)

        # Añadir campos para max_cost, min_resistance y min_durability
        self.label_max_cost = QLabel('Costo máximo (USD):')
        self.input_max_cost = QLineEdit()
        layout.addWidget(self.label_max_cost)
        layout.addWidget(self.input_max_cost)

        self.label_min_resistance = QLabel('Resistencia mínima (MPa):')
        self.input_min_resistance = QLineEdit()
        layout.addWidget(self.label_min_resistance)
        layout.addWidget(self.input_min_resistance)

        self.label_min_durability = QLabel('Durabilidad mínima (años):')
        self.input_min_durability = QLineEdit()
        layout.addWidget(self.label_min_durability)
        layout.addWidget(self.input_min_durability)

        self.label_subtitulo_operacion = QLabel('Datos de la operación')
        layout.addWidget(self.label_subtitulo_operacion)

        self.button_iniciar = QPushButton('Start')
        layout.addWidget(self.button_iniciar)

        # Agregar un QTextEdit para mostrar la información de la población
        self.population_info = QTextEdit()
        self.population_info.setReadOnly(True)
        layout.addWidget(self.population_info)

        self.setLayout(layout)
        self.button_iniciar.clicked.connect(self.start_genetic_algorithm)

    
    def start_genetic_algorithm(self):
      self.population_info.clear()
      initial_value = float(self.input_inicial.text())
      final_value = float(self.input_final.text())
      generations = int(self.input_generaciones.text())
      individual_mutation = float(self.input_mutacion_individual.text())
      mutation_per_gen = float(self.input_mutacion_por_gen.text())
      min_resistance = float(self.input_min_resistance.text())
      min_durability = float(self.input_min_durability.text())

      algorithm = GeneticAlgorithm(initial_value, final_value, generations, individual_mutation, mutation_per_gen, max_cost, min_resistance, min_durability)
      best_individuals, properties_values = algorithm.run()

      for i, properties in enumerate(properties_values):
          info_text = f"""
          Generación {i + 1}:
          - Costo (USD/kg): {properties['Costo (USD/kg)']}
          - Resistencia (MPa): {properties['Resistencia (MPa)']}
          - Durabilidad (años): {properties['Durabilidad (años)']}
          - Peso (kg/m³): {properties['Peso (kg/m³)']}
          - Resistencia a la Corrosión (%): {properties['Resistencia a la Corrosión (%)']}
          - Absorción de Agua (%): {properties['Absorción de Agua (%)']}
          - Conductividad Térmica (W/m·K): {properties['Conductividad Térmica (W/m·K)']}
          """
          self.population_info.append(info_text)

    # Mostrar los 3 mejores individuos de la última generación
      self.population_info.append("\nLos 3 mejores individuos")
      best_individuals.sort(key=algorithm.fitness)
      unique_best_individuals = []
      for individual in best_individuals:
          if individual not in unique_best_individuals:
              unique_best_individuals.append(individual)
          if len(unique_best_individuals) == 3:
              break

      for i, individual in enumerate(unique_best_individuals):
          best_ind_text = f"Individuo {i + 1}: {individual}"
          best_ind_text += f", Error: {algorithm.fitness(individual)}"
          self.population_info.append(best_ind_text)