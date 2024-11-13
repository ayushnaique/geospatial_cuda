from tqdm import tqdm
import random

def generate_random_points(filename, num_points, grid_size=(1e6, 1e6)):
    with open(filename, "w") as file:
        for _ in tqdm(range(int(num_points))):
            x = random.randint(0, grid_size[0] - 1)
            y = random.randint(0, grid_size[1] - 1)
            file.write(f"{x} {y}\n")

def main():
    # Approximate maximum points to keep file size under 1 GB
    max_points = 1e6
    filename = "points.txt"
    
    generate_random_points(filename, max_points)
    print(f"Generated {max_points} random points in {filename}")

if __name__ == "__main__":
    main()