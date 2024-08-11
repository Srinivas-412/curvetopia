!pip install numpy scipy svgwrite cairosvg


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import svgwrite
import base64
from scipy.spatial.distance import cdist
from google.colab import files

# Function to read CSV
def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

# Function to fit Bezier curve
def bezier_curve_fit(points):
    def bezier(t, p0, p1, p2, p3):
        return ((1-t)**3)[:, None]*p0 + 3*((1-t)**2)[:, None]*t[:, None]*p1 + 3*(1-t)[:, None]*(t**2)[:, None]*p2 + (t**3)[:, None]*p3

    def objective(params, points):
        num_points = len(points)
        if len(params) != 4 * 2:  # 4 points * 2 coordinates each
            raise ValueError("The length of params does not match the expected number of control points.")

        p0, p1, p2, p3 = np.split(params, 4)
        p0 = p0.reshape(2)
        p1 = p1.reshape(2)
        p2 = p2.reshape(2)
        p3 = p3.reshape(2)
        t = np.linspace(0, 1, len(points))
        curve = bezier(t, p0, p1, p2, p3)
        return np.mean(np.linalg.norm(points - curve, axis=1))

    if len(points) < 4:
        raise ValueError("Insufficient number of points to fit a Bezier curve.")

    initial_params = np.concatenate([
        points[0],
        points[len(points) // 3],
        points[2 * len(points) // 3],
        points[-1]
    ])
    initial_params = initial_params.flatten()  # Ensure initial_params is 1-dimensional
    result = minimize(objective, initial_params, args=(points,), method='BFGS')
    return np.split(result.x, 4)

# Function to create SVG path data
def bezier_path_data(control_points):
    p0, p1, p2, p3 = [point.reshape(2) for point in control_points]
    return f'M {p0[0]} {p0[1]} C {p1[0]} {p1[1]}, {p2[0]} {p2[1]}, {p3[0]} {p3[1]}'

# Function to create SVG file
def create_svg(polylines, filename):
    dwg = svgwrite.Drawing(filename, profile='tiny')
    for polyline in polylines:
        if np.array(polyline).ndim != 2:
            print(f"Skipping invalid polyline with ndim: {np.array(polyline).ndim}")
            continue
        points = np.array(polyline)
        try:
            control_points = bezier_curve_fit(points)
            path_data = bezier_path_data(control_points)
            print(f"Adding path to SVG: {path_data}")  # Debug: Print the SVG path data
            dwg.add(dwg.path(d=path_data, stroke=svgwrite.rgb(0, 0, 0, '%'), fill='none'))
        except ValueError as e:
            print(f"Error in fitting Bezier curve: {e}")
    dwg.save()

# Function to create HTML file to display SVG
def create_html(html_filename, svg_filename):
    # Encode SVG file as base64
    with open(svg_filename, "rb") as svg_file:
        svg_data = svg_file.read()
    svg_base64 = base64.b64encode(svg_data).decode('utf-8')

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVG Display</title>
</head>
<body>
    <h1>SVG Display</h1>
    <!-- Embed the SVG file using base64 encoding -->
    <img src="data:image/svg+xml;base64,{svg_base64}" width="600" height="600" alt="SVG Image">
</body>
</html>
    """
    with open(html_filename, 'w') as file:
        file.write(html_content)

# Function to plot polylines
def plot(paths_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.show()

# Function to check for symmetry
def is_symmetric(points, tol=1e-2):
    center = np.mean(points, axis=0)
    dists = cdist(points, [center])
    return np.all(np.abs(dists - np.mean(dists)) < tol)

# Function to regularize curves (e.g., fitting to known shapes like circles)
def regularize_curve(points):
    if points.ndim != 2 or points.shape[1] != 2:
        print("Skipping invalid polyline with shape:", points.shape)
        return points
    if is_symmetric(points):
        center = np.mean(points, axis=0)
        radius = np.mean(np.linalg.norm(points - center, axis=1))
        angles = np.linspace(0, 2 * np.pi, len(points))
        return center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    return points

# Function to complete incomplete curves
def complete_curve(points, num_points=10):
    if len(points) < 2:
        return points
    new_points = []
    for i in range(len(points) - 1):
        new_points.append(points[i])
        for j in range(1, num_points):
            t = j / num_points
            new_point = (1 - t) * np.array(points[i]) + t * np.array(points[i+1])
            new_points.append(new_point.tolist())
    new_points.append(points[-1])
    return new_points

# Upload the CSV file
uploaded = files.upload()
csv_path = list(uploaded.keys())[0]

# Read the CSV file
polylines = read_csv(csv_path)

# Apply regularization and symmetry detection
regularized_polylines = []
for polyline in polylines:
    try:
        regularized_polylines.append(regularize_curve(np.array(polyline)))
    except Exception as e:
        print(f"Error in regularizing polyline: {e}")

# Complete the curves (fill gaps)
completed_polylines = [complete_curve(polyline) for polyline in regularized_polylines]

# Plot the polylines (for debugging purposes)
plot(completed_polylines)

# Create SVG file
svg_filename = 'output.svg'
create_svg(completed_polylines, svg_filename)

# Create HTML file to display the SVG
html_filename = 'index.html'
create_html(html_filename, svg_filename)

# Download the HTML file
files.download(html_filename)
