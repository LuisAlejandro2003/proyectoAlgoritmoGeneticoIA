import pandas as pd

# Datos de materiales
data = {
    'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    'Material': [
        'Cemento Portland', 'Arena', 'Grava', 'Agua', 'Silica Fume', 'Cal', 'Cenizas Volantes', 'Escoria Granulada',
        'Aditivo Reductor de Agua', 'Polímero Superplastificante', 'Cemento de Aluminio', 'Vidrio Molido', 
        'Polvo de Mármol', 'Polvo de Piedra Caliza', 'Fibra de Vidrio', 'Polipropileno', 'Acero de Refuerzo', 
        'Fibra de Carbono', 'Polvo de Granito', 'Látex', 'Bentonita', 'Yeso', 'Vermiculita', 'Celulosa', 
        'Fibra de Poliéster', 'Mica', 'Escoria de Alto Horno', 'Bentonita Sódica', 'Cuarzo', 'Perlita'
    ],
    'Resistencia (MPa)': [40, 5, 10, 0, 70, 8, 20, 15, 5, 100, 70, 35, 25, 15, 150, 35, 500, 200, 60, 10, 10, 7, 1, 0.5, 50, 35, 30, 12, 80, 2],
    'Costo (USD/kg)': [0.12, 0.02, 0.03, 0.001, 0.6, 0.05, 0.02, 0.03, 0.4, 0.8, 0.15, 0.05, 0.1, 0.04, 1, 2, 0.6, 8, 0.05, 3, 0.03, 0.015, 0.25, 0.4, 1.5, 0.03, 0.04, 0.035, 0.07, 0.1],
    'Durabilidad (años)': [50, 100, 150, None, 60, 70, 80, 90, 40, 30, 40, 70, 90, 100, 20, 30, 100, 40, 50, 10, 120, 30, 80, 25, 15, 150, 70, 110, 200, 50],
    'Peso (kg/m³)': [1440, 1600, 1800, 1000, 2200, 2100, 700, 2900, 1100, 900, 3200, 2500, 2700, 2400, 2600, 900, 7850, 1800, 2750, 950, 800, 950, 1000, 1200, 1450, 2700, 3300, 950, 2650, 1100],
    'Resistencia a la Corrosión (%)': [80, 90, 95, None, 85, 75, 65, 88, 70, 80, 90, 85, 80, 75, 95, 85, 100, 90, 85, 70, 85, 80, 65, 70, 75, 90, 85, 75, 95, 65],
    'Absorción de Agua (%)': [6, 1, 0.5, 0, 0.1, 2, 1.2, 0.6, 0.1, 0.01, 4, 0.2, 0.3, 0.4, 0.2, 0.01, 0.2, 0.1, 0.4, 0.05, 5, 2, 5, 6, 0.3, 0.5, 0.4, 5.5, 0.1, 3],
    'Conductividad Térmica (W/m·K)': [0.9, 0.2, 1, 0.6, 1.5, 0.6, 0.4, 0.3, 0.8, 0.25, 1.2, 1.5, 2.1, 2, 0.8, 0.2, 50, 0.6, 3, 0.2, 0.5, 0.3, 0.12, 0.05, 0.4, 2.2, 0.9, 0.6, 6, 0.3]
}

# Crear un DataFrame
df = pd.DataFrame(data)

# Guardar en un archivo CSV
df.to_csv('materiales_construccion_grande.csv', index=False)

print("Dataset guardado como 'materiales_construccion_grande.csv'.")
