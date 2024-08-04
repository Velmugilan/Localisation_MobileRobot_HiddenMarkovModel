import pygame
import sys
import math
import csv
import random

# Initialize Pygame
pygame.init()

# Set the width and height of the screen (500x500)
screen_width = 500
screen_height = 500
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("2D Turtle Bot with LiDAR Sensor")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
RED = (255, 0, 0)

# Turtle Bot
turtle_size = 20
turtle_radius = turtle_size // 2
turtle_x = 25
turtle_y = 25

# Movement speed
speed = 50

# Obstacles
obstacles = []

# Define obstacle positions in a particular grid
obstacle_positions = [
    (200, 0), (0, 50), (50, 50), (100, 50), (200, 50), (300, 50), (350, 50), (400, 50), (450, 50), (100, 100), (200, 100), (100, 150), (200, 150), (250, 150), (300, 150), (350, 150), (100, 200),
    (200, 200), (250, 200), (300, 200), (350, 200), (0, 300), (50, 300), (150, 300), (200, 300), (250, 300), (300, 300), (350, 300), (400, 300), (450, 300), (0, 350), (50, 350), (0, 400),
    (50, 400), (100, 400), (150, 400), (300, 400), (350, 400), (400, 400), (0, 450), (50, 450)
]

# Convert obstacle positions to Rect objects
for pos in obstacle_positions:
    obstacle_rect = pygame.Rect(pos[0], pos[1], 50, 50)  # Adjust the size as needed
    obstacles.append(obstacle_rect)

# LiDAR Sensor
sensor_range = 100
sensor_angles = [math.radians(angle) for angle in range(0, 360)]
lidar_values = []

# Data for CSV
csv_data = []

# Tower positions
tower_positions = [(0, 0), (0, 500), (500, 0), (500, 500)]

def update_lidar_values():
    lidar_values.clear()
    for angle in sensor_angles:
        end_x = turtle_x + (turtle_radius + sensor_range) * math.cos(angle)
        end_y = turtle_y + (turtle_radius + sensor_range) * math.sin(angle)
        min_distance = sensor_range
        for obstacle in obstacles:
            obstacle_center_x = obstacle.x + obstacle.width / 2
            obstacle_center_y = obstacle.y + obstacle.height / 2
            distance = math.sqrt((turtle_x - obstacle_center_x) ** 2 + (turtle_y - obstacle_center_y) ** 2) - turtle_radius
            noisy_sensor_distance = distance + random.uniform(-10, 10)
            if noisy_sensor_distance < min_distance:
                min_distance = noisy_sensor_distance
        lidar_values.append(0 if min_distance > sensor_range else sensor_range - min_distance)


def draw_grid():
    for x in range(0, screen_width, screen_width // 10):
        pygame.draw.line(screen, BLACK, (x, 0), (x, screen_height))
    for y in range(0, screen_height, screen_height // 10):
        pygame.draw.line(screen, BLACK, (0, y), (screen_width, y))


# Main loop
running = True
while running:
    screen.fill(WHITE)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get the keys pressed
    keys = pygame.key.get_pressed()

    draw_grid()

    # Attempt to move the turtle bot based on the keys pressed
    new_turtle_x = turtle_x
    new_turtle_y = turtle_y
    if keys[pygame.K_LEFT]:
        new_turtle_x -= speed
    if keys[pygame.K_RIGHT]:
        new_turtle_x += speed
    if keys[pygame.K_UP]:
        new_turtle_y -= speed
    if keys[pygame.K_DOWN]:
        new_turtle_y += speed

    # Boundaries check for turtle bot
    new_turtle_x = max(turtle_radius, min(screen_width - turtle_radius, new_turtle_x))
    new_turtle_y = max(turtle_radius, min(screen_height - turtle_radius, new_turtle_y))

    # Check for collision with obstacles before moving
    turtle_rect = pygame.Rect(new_turtle_x - turtle_radius, new_turtle_y - turtle_radius, turtle_size, turtle_size)
    collision = False
    for obstacle in obstacles:
        if turtle_rect.colliderect(obstacle):
            collision = True
            break

    # Move the turtle bot only if there's no collision
    if not collision:
        turtle_x = new_turtle_x
        turtle_y = new_turtle_y

    # Update LiDAR values
    update_lidar_values()

    # Draw the obstacles
    for obstacle in obstacles:
        pygame.draw.rect(screen, GRAY, obstacle)

    # Draw LiDAR sensor lines

    for i, angle in enumerate(sensor_angles):
        end_x = turtle_x + (turtle_radius + 100 - lidar_values[i]) * math.cos(angle)
        end_y = turtle_y + (turtle_radius + 100 - lidar_values[i]) * math.sin(angle)
        pygame.draw.line(screen, RED, (turtle_x, turtle_y), (end_x, end_y))

    # Draw the turtle bot
    pygame.draw.circle(screen, BLACK, (turtle_x, turtle_y), turtle_radius)

    # Draw the towers
    for tower_pos in tower_positions:
        pygame.draw.circle(screen, BLACK, tower_pos, 5)

    # Calculate distances to towers with noise
    tower_distances = []
    for tower_pos in tower_positions:
        distance = math.sqrt(((turtle_x - tower_pos[0])/50) ** 2 + ((turtle_y - tower_pos[1])/50) ** 2)
        noisy_distance = distance + random.uniform(-0.2, 0.2)  # Add random noise between -5 and 5
        tower_distances.append(round(noisy_distance, 2))  # Round to 2 decimal places

    print(tower_distances)

    # Display real-time position, corresponding sensor value, and tower distances
    font = pygame.font.SysFont(None, 24)
    text = font.render(f"Position: ({turtle_x}, {turtle_y}), Sensor Value: {min(lidar_values):.2f}, Tower Distances: {tower_distances}", True, BLACK)
    screen.blit(text, (10, 10))

    # Store data for CSV
    csv_data.append([turtle_x, turtle_y, min(lidar_values)  ] + tower_distances)

    # Update the display
    pygame.display.flip()

    # Control the speed of the game
    pygame.time.Clock().tick(4)

# Write data to CSV file
with open("robot_data.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["X", "Y", "Sensor Value", "Tower 1 Distance", "Tower 2 Distance", "Tower 3 Distance", "Tower 4 Distance"])
    writer.writerows(csv_data)

# Quit Pygame
pygame.quit()

sys.exit()
