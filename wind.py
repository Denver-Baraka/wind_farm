import numpy as np

class WindFarm:
    def __init__(self, wind_speed, rotor_diameter, hub_height, rated_power, power_curve, generator_efficiency, air_density, thrust_coefficient, num_turbines, turbine_layout):
        self.wind_speed = wind_speed
        self.rotor_diameter = rotor_diameter
        self.hub_height = hub_height
        self.rated_power = rated_power
        self.power_curve = power_curve
        self.generator_efficiency = generator_efficiency
        self.air_density = air_density
        self.thrust_coefficient = thrust_coefficient
        self.num_turbines = num_turbines
        self.turbine_layout = turbine_layout
        self.wake_model_params = {
            'kw': 0.04  # Wake decay constant (can be adjusted based on site-specific data)
        }

    def jensen_wake_deficit(self, distance):
        kw = self.wake_model_params['kw']
        wake_deficit = (1 - np.sqrt(1 - self.thrust_coefficient)) / (1 + kw * distance / self.rotor_diameter)**2
        return wake_deficit

    def calculate_effective_wind_speed(self):
        effective_wind_speeds = np.full(self.num_turbines, self.wind_speed)
        for i in range(self.num_turbines):
            for j in range(i):
                distance = np.linalg.norm(np.array(self.turbine_layout[i]) - np.array(self.turbine_layout[j]))
                wake_deficit = self.jensen_wake_deficit(distance)
                effective_wind_speeds[i] -= self.wind_speed * wake_deficit
        return effective_wind_speeds

    def power_output(self, wind_speed):
        swept_area = np.pi * (self.rotor_diameter / 2)**2
        power_output = 0.5 * self.air_density * swept_area * self.power_curve(wind_speed) * self.generator_efficiency
        return power_output

    def total_power_output(self):
        effective_wind_speeds = self.calculate_effective_wind_speed()
        total_power = sum(self.power_output(ws) for ws in effective_wind_speeds)
        return total_power

def power_curve(wind_speed):
    # A simple example power curve function
    if wind_speed < 3:
        return 0
    elif 3 <= wind_speed < 12:
        return (wind_speed - 3) / 9 * rated_power
    elif 12 <= wind_speed < 25:
        return rated_power
    else:
        return 0

# Request user input
wind_speed = float(input("Enter the wind speed (m/s): "))
rotor_diameter = float(input("Enter the rotor diameter (m): "))
hub_height = float(input("Enter the hub height (m): "))
rated_power = float(input("Enter the rated power of the turbine (kW): "))
num_turbines = int(input("Enter the number of turbines: "))
spacing = float(input("Enter the spacing between turbines (multiples of rotor diameter): "))

# Generate a default grid layout
grid_size = int(np.ceil(np.sqrt(num_turbines)))  # Determine the size of the grid
turbine_layout = [(i * spacing * rotor_diameter, j * spacing * rotor_diameter) for i in range(grid_size) for j in range(grid_size)][:num_turbines]

# Define default parameters
generator_efficiency = 0.9  # 90%
air_density = 1.225  # kg/m^3 (standard at sea level)
thrust_coefficient = 0.8

# Create the wind farm model
wind_farm = WindFarm(wind_speed, rotor_diameter, hub_height, rated_power, power_curve, generator_efficiency, air_density, thrust_coefficient, num_turbines, turbine_layout)

# Calculate total power output
total_power = wind_farm.total_power_output()
print(f"Total Power Output of the Wind Farm: {total_power / 1000:.2f} KW")
