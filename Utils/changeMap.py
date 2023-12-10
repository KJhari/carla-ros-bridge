import carla

# Connect to the CARLA server running on localhost:2000
client = carla.Client('localhost', 2000)
client.set_timeout(10.0) # seconds

# Change the map
map_name = 'Town07_Opt'  # Replace with the map you want to load
#map_name = 'Town02_Opt'  # Replace with the map you want to load

world = client.load_world(map_name)

# Toggle all buildings off
world.unload_map_layer(carla.MapLayer.Buildings)
