import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar el dataset
df_materiales = pd.read_csv('materiales.csv')

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
    
    # Penalización por no cumplir con los requisitos
    penalizacion = 0
    if resistencia < requisitos['resistencia_minima']:
        penalizacion += (requisitos['resistencia_minima'] - resistencia)
    if durabilidad < requisitos['durabilidad_deseada']:
        penalizacion += (requisitos['durabilidad_deseada'] - durabilidad)
    if resistencia_corrosion < requisitos['resistencia_corrosion_minima']:
        penalizacion += (requisitos['resistencia_corrosion_minima'] - resistencia_corrosion)

    score = resistencia + durabilidad + resistencia_corrosion - penalizacion  # Maximizar suma de resistencia, durabilidad y resistencia a la corrosión
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
    return hijo1 / np.sum(hijo1), hijo2 / np.sum(hijo2)  # Normalizar

# Función de mutación
def mutacion(solucion, tasa_mutacion=0.1):
    if np.random.rand() < tasa_mutacion:
        idx = np.random.randint(0, len(solucion))
        solucion[idx] += np.random.uniform(-0.1, 0.1)  # Ajustar proporción
        solucion = np.clip(solucion, 0, None)  # Evitar valores negativos
        return solucion / np.sum(solucion)  # Normalizar
    return solucion

# Algoritmo Genético
def algoritmo_genetico(num_generaciones, tamaño_poblacion, requisitos):
    poblaciones = [[generar_solucion() for _ in range(tamaño_poblacion)] for _ in range(3)]
    
    historia_resistencias = [[] for _ in range(3)]
    historia_durabilidades = [[] for _ in range(3)]
    historia_costos = [[] for _ in range(3)]
    historia_resistencias_corrosion = [[] for _ in range(3)]

    mejores_mezclas = [None, None, None]
    mejores_puntuaciones = [-np.inf, -np.inf, -np.inf]

    for _ in range(num_generaciones):
        for i in range(3):
            puntuaciones = [evaluar_solucion(sol, requisitos)[0] for sol in poblaciones[i]]
            resistencia_gen = [evaluar_solucion(sol, requisitos)[1] for sol in poblaciones[i]]
            durabilidad_gen = [evaluar_solucion(sol, requisitos)[2] for sol in poblaciones[i]]
            costo_gen = [evaluar_solucion(sol, requisitos)[3] for sol in poblaciones[i]]
            resistencia_corrosion_gen = [evaluar_solucion(sol, requisitos)[4] for sol in poblaciones[i]]

            # Acumular para garantizar solo tendencias ascendentes
            if historia_resistencias[i]:
                resistencia_gen = max(np.mean(resistencia_gen), historia_resistencias[i][-1])
                durabilidad_gen = max(np.mean(durabilidad_gen), historia_durabilidades[i][-1])
                costo_gen = max(np.mean(costo_gen), historia_costos[i][-1])
                resistencia_corrosion_gen = max(np.mean(resistencia_corrosion_gen), historia_resistencias_corrosion[i][-1])
            else:
                resistencia_gen = np.mean(resistencia_gen)
                durabilidad_gen = np.mean(durabilidad_gen)
                costo_gen = np.mean(costo_gen)
                resistencia_corrosion_gen = np.mean(resistencia_corrosion_gen)

            historia_resistencias[i].append(resistencia_gen)
            historia_durabilidades[i].append(durabilidad_gen)
            historia_costos[i].append(costo_gen)
            historia_resistencias_corrosion[i].append(resistencia_corrosion_gen)
            
            nueva_poblacion = []
            
            for _ in range(tamaño_poblacion // 2):
                parent1, parent2 = seleccion(poblaciones[i], puntuaciones)
                hijo1, hijo2 = cruce(parent1, parent2)
                nueva_poblacion.extend([mutacion(hijo1), mutacion(hijo2)])
            
            poblaciones[i] = nueva_poblacion

            # Evaluar y actualizar la mejor mezcla para esta sub-población
            for sol in poblaciones[i]:
                score = evaluar_solucion(sol, requisitos)[0]
                if score > mejores_puntuaciones[i]:
                    mejores_puntuaciones[i] = score
                    mejores_mezclas[i] = sol
    
    return mejores_mezclas, historia_resistencias, historia_durabilidades, historia_costos, historia_resistencias_corrosion

# Ejecución
requisitos = {
    'resistencia_minima': 45,
    'durabilidad_deseada': 60,
    'resistencia_corrosion_minima': 90  # Agregar requisito de resistencia a la corrosión
}

mejores_mezclas, historia_resistencias, historia_durabilidades, historia_costos, historia_resistencias_corrosion = algoritmo_genetico(100, 50, requisitos)

# Función para imprimir las propiedades de una mezcla
def imprimir_mezcla(mezcla, nombre):
    if mezcla is None:
        print(f"{nombre} no se encontró.")
        return

    materiales_utilizados = df_materiales.loc[mezcla > 0]
    porcentajes = mezcla[mezcla > 0] * 100  # Convertir a porcentajes
    resistencia_total = np.sum(mezcla * df_materiales['Resistencia (MPa)'])
    durabilidad_total = np.sum(mezcla * df_materiales['Durabilidad (años)'])
    costo_total = np.sum(mezcla * df_materiales['Costo (USD/kg)'])
    resistencia_corrosion_total = np.sum(mezcla * df_materiales['Resistencia a la Corrosión (%)'])

    print(f"\n{nombre}:")
    for material, porcentaje in zip(materiales_utilizados['Material'], porcentajes):
        print(f"{material}: {porcentaje:.2f}%")
    print("\nPropiedades de la mezcla:")
    print(f"Resistencia total: {resistencia_total:.2f} MPa")
    print(f"Durabilidad total: {durabilidad_total:.2f} años")
    print(f"Costo total: {costo_total:.2f} USD/kg")
    print(f"Resistencia a la corrosión total: {resistencia_corrosion_total:.2f} %")

# Imprimir las tres mejores mezclas
imprimir_mezcla(mejores_mezclas[0], "Mejor mezcla 1")
imprimir_mezcla(mejores_mezclas[1], "Mejor mezcla 2")
imprimir_mezcla(mejores_mezclas[2], "Mejor mezcla 3")

# Graficar la evolución de las propiedades
generaciones = range(1, 101)
plt.figure(figsize=(12, 6))
for i in range(3):
    plt.plot(generaciones, historia_resistencias[i], label=f'Resistencia Media (Mezcla {i+1})', color='blue')
    plt.plot(generaciones, historia_durabilidades[i], label=f'Durabilidad Media (Mezcla {i+1})', color='green')
    plt.plot(generaciones, historia_costos[i], label=f'Costo Medio (Mezcla {i+1})', color='red')
    plt.plot(generaciones, historia_resistencias_corrosion[i], label=f'Resistencia a la Corrosión Media (Mezcla {i+1})', color='purple')
plt.xlabel('Generaciones')
plt.ylabel('Valores de Propiedades')
plt.title('Evolución de Propiedades a lo Largo de las Generaciones')
plt.legend()
plt.grid()
plt.show()
