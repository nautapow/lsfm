import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import math
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

import libpysal
from esda.moran import Moran
import seaborn as sns
%matplotlib inline
# =============================================================================
# def transform_img(coords, points):
#     ax, ay = coords[0][0], coords[0][1]
#     bx, by = coords[1][0], coords[1][1]
#     cx, cy = coords[2][0], coords[2][1]
#     dx, dy = coords[3][0], coords[3][1]
# 
#     B_ori = np.array([bx, by])
#     C_ori = np.array([cx, cy])
#     D_ori = np.array([dx, dy])
# 
#     A = np.array([ax-ax,ay-ay])
#     B = np.array([bx-ax,by-ay])
#     C = np.array([cx-ax,cy-ay])
#     D = np.array([dx-ax,dy-ay])
# 
#     
#     u = D-A
#     unit = np.linalg.norm(u).astype('float64')
#     
#     A2UF = B-C
#     v = (A2UF/np.linalg.norm(A2UF)) * unit
#     
#     basis = np.column_stack((u, v))
#     basis_inv = np.linalg.inv(basis)
#     
#     # Transform all points
#     def new_coord(p):
#         p_shift = p - np.array([ax, ay])
#         coeffs = basis_inv @ p_shift
#         return tuple(np.round(coeffs, 3))
#     
#     new_points = [new_coord(np.array([px, py])) for px, py in points]
#     
#     corners = [(0,0),
#                new_coord(B_ori),
#                new_coord(C_ori),
#                new_coord(D_ori)]
#     
#     return corners, new_points
# =============================================================================

# =============================================================================
# #axis = A1Low-AAFLow, A2High-A1High
# def transform_img(coords, points):
#     def line_intersection(p1, p2, p3, p4):
#         """
#         Returns intersection point of lines p1p2 and p3p4 (if they intersect)
#         p1, p2, p3, p4 are (x, y) tuples or arrays
#         """
#         A = np.array(p1, dtype=float)
#         B = np.array(p2, dtype=float)
#         C = np.array(p3, dtype=float)
#         D = np.array(p4, dtype=float)
#     
#         # Vectors along the lines
#         v1 = B - A
#         v2 = D - C
#         w = A - C
#     
#         # Solve: A + t*v1 = C + s*v2
#         matrix = np.column_stack((v1, -v2))
#         if np.linalg.matrix_rank(matrix) < 2:
#             raise ValueError("Lines are parallel and do not intersect")
#     
#         params = np.linalg.solve(matrix, -w)
#         t = params[0]
#     
#         # Intersection point
#         intersection = A + t * v1
#         
#         return intersection
#     
#     A = np.array(coords[0])
#     B = np.array(coords[1])
#     C = np.array(coords[2])
#     D = np.array(coords[3])
#     
#     intersec = line_intersection(A,D,C,B)
#     vec_AD = D - A
#     vec_BC = B - C
#     
#     u = vec_AD / np.linalg.norm(vec_AD)
#     v = vec_BC / np.linalg.norm(vec_BC)
#     
#     basis = np.column_stack((u, v))
#     basis_inv = np.linalg.inv(basis)
#     
#     # Transform all points
#     def new_coord(p):
#         p_shift = p - intersec
#         coeffs = basis_inv @ p_shift
#         scale = (coeffs[0]/np.linalg.norm(vec_AD), coeffs[1]/np.linalg.norm(vec_BC))
#         return tuple(np.round(scale, 3))
#     
#     new_coords = [new_coord(A), new_coord(B), new_coord(C), new_coord(D)]
#     new_points = [new_coord(np.array([px, py])) for px, py in points]
#     
#     return new_coords, new_points
# =============================================================================

#Axis A1Low-A2High, MidA1A2(Perpendicular)
def transform_img(coords, points):
    A = np.array(coords[0])
    B = np.array(coords[1])
    C = np.array(coords[2])
    D = np.array(coords[3])
    
    vec_AC = C - A
    mid_AC = (A + C) / 2
    
    u = vec_AC / np.linalg.norm(vec_AC)
    v = np.array([u[1], -u[0]])
    
    
    R = np.stack((u, v), axis=1)
    
    # Transform all points
    def new_coord(p):
        p_shift = p - mid_AC
        p_rot = R.T @ p_shift
        
        return p_rot/np.linalg.norm(vec_AC)
    
    
    new_coords = [new_coord(A), new_coord(B), new_coord(C), new_coord(D)]
    new_points = [new_coord(np.array([px, py])) for px, py in points]
    
    return new_coords, new_points



# =============================================================================
# def transform_img(coords, points):
#     ax, ay = coords[0]
#     bx, by = coords[1]
#     cx, cy = coords[2]
#     dx, dy = coords[3]
# 
#     A = np.array([ax, ay])
#     B = np.array([bx, by])
#     C = np.array([cx, cy])
#     D = np.array([dx, dy])
# 
#     # Step 1: Find intersection point P of lines AD and BC
#     P = line_intersection(A, D, B, C)
# 
#     # Step 2: Define basis vectors (AD and BC directions from P)
#     vec_AD = D - A
#     vec_BC = C - B
# 
#     u = vec_AD / np.linalg.norm(vec_AD)
#     v = vec_BC / np.linalg.norm(vec_BC)
# 
#     # Optional: scale v to same length as u (if desired, like in your old code)
#     unit = np.linalg.norm(vec_AD)
#     v = v * unit
# 
#     basis = np.column_stack((u, v))
#     basis_inv = np.linalg.inv(basis)
# 
#     # Transform all points relative to P
#     def new_coord(p):
#         p_shift = p - P
#         coeffs = basis_inv @ p_shift
#         return tuple(np.round(coeffs, 3))
# 
#     new_points = [new_coord(np.array([px, py])) for px, py in points]
# 
#     corners = [new_coord(A),
#                new_coord(B),
#                new_coord(C),
#                new_coord(D)]
# 
#     return corners, new_points
# =============================================================================



def transform_img_3p(coords, points):
    ax, ay = coords[0][0], coords[0][1]
    bx, by = coords[1][0], coords[1][1]
    cx, cy = coords[2][0], coords[2][1]
    dx, dy = coords[3][0], coords[3][1]

    
    A = np.array([ax-ax,ay-ay])
    B = np.array([bx-ax,by-ay])
    C = np.array([cx-ax,cy-ay])
    D = np.array([dx-ax,dy-ay])

    
    u = B-A
    unit = np.linalg.norm(u).astype('float64')

    am = C-A
    v = (am/np.linalg.norm(am)) * unit
    
    basis = np.column_stack((u, v))
    basis_inv = np.linalg.inv(basis)
    
    
    # Transform all points
    def new_coord(p):
        p_shift = p - np.array([ax, ay])
        coeffs = basis_inv @ p_shift
        return tuple(np.round(coeffs, 3))
    
    new_points = [new_coord(np.array([px, py])) for px, py in points]
    
    corners = [(0,0),
               new_coord(B),
               new_coord(C),
               new_coord(D)]
    
    return corners, new_points


if __name__ == '__main__':
    coords = pd.read_excel('mapping_coordinate.xlsx', sheet_name='coords')
    tone_info = pd.read_excel('tone_cell_note.xlsx')
    mouseid = list(tone_info['mouseID'])
    filename = list(tone_info['filename'])
    sites = list(tone_info['patch_site'])
    
    patch_x, patch_y =[],[]
    for i,m in enumerate(mouseid):
        site = f'Patch_{sites[i]}'
        patch_x.append(coords[(coords['mouseid'] == m) & (coords['regions'] == site)].x_A12.item())
        patch_y.append(coords[(coords['mouseid'] == m) & (coords['regions'] == site)].y_A12.item())
        
    bf = np.array(list(tone_info['best_frequency']))/1000
    
    df = pd.DataFrame({'x': patch_x, 'y': patch_y, 'frequency': bf})
    
# =============================================================================
#     # === 3. K-means clustering on frequency ===
#     kmeans = KMeans(n_clusters=2, random_state=0)
#     df['freq_group'] = kmeans.fit_predict(df[['frequency']])
#     
#     # Optional: Label groups as 'low' and 'high' based on mean
#     group_means = df.groupby('freq_group')['frequency'].mean()
#     high_group = group_means.idxmax()
#     df['freq_group_label'] = df['freq_group'].apply(lambda g: 'high' if g == high_group else 'low')
# =============================================================================
    
# =============================================================================
#     median = df['frequency'].median()
#     df['freq_group_label'] = df['frequency'].apply(lambda v: 'high' if v > median else 'low')
# =============================================================================
    
    
    ## two methods to separate frequency into three groups
    df['freq_group_label'] = pd.qcut(df['frequency'], q=3, labels=['low', 'medium', 'high'])

    # Check counts
    print(df['freq_group_label'].value_counts())
    
    
    from scipy.stats import mannwhitneyu

    # Split groups
    high_group = df[df['freq_group_label'] == 'high']['frequency']
    medium_group = df[df['freq_group_label'] == 'medium']['frequency']
    low_group = df[df['freq_group_label'] == 'low']['frequency']
    
    print(f'avg frequency of each group: {np.mean(high_group)}, {np.mean(medium_group)}, {np.mean(low_group)}')
    
    # Mann-Whitney U test
    u_stat, p_value = mannwhitneyu(low_group, medium_group, alternative='two-sided')
    
    print(f"Mann–Whitney U statistic: {u_stat}")
    print(f"p-value: {p_value:.7f}")
    
    
    # === 4. Plot coordinates with freq group ===
    # Add jitter to avoid overlap
    jitter_strength = 0.05
    np.random.seed(42)
    df['x_jitter'] = df['x'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(df))
    df['y_jitter'] = df['y'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(df))
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='x_jitter', y='y_jitter', hue='freq_group_label', palette='Set1', s=80)
    plt.title('Coordinates colored by Frequency Group (median)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(title='Frequency Group')
    plt.show()
    
    # === 5. Moran's I on freq group ===
    # First, create spatial weights (e.g., K-nearest neighbors)
    coords = list(zip(df['x_jitter'], df['y_jitter']))
    knn = libpysal.weights.KNN.from_array(coords, k=5)  # k=4 nearest neighbors (adjustable)
    knn.transform = 'r'  # row-standardize weights
    
    # Convert 'high'/'low' to binary (1=high, 0=low)
    df['freq_binary'] = (df['freq_group_label'] == 'high').astype(int)
    
    # Calculate Moran's I
    moran = Moran(df['freq_binary'], knn)
    
    print(f"Moran’s I: {moran.I:.4f}")
    print(f"p-value (permutation test): {moran.p_sim:.4f}")
    
    # === 6. DBSCAN clustering on coordinates ===
    # Standardize coordinates (important for DBSCAN)
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(df[['x_jitter', 'y_jitter']])
    
    dbscan = DBSCAN(eps=0.7, min_samples=3)  # eps & min_samples need tuning!
    df['dbscan_cluster'] = dbscan.fit_predict(coords_scaled)
    
    # === 7. Plot DBSCAN clusters vs Frequency group ===
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='x_jitter', y='y_jitter', hue='dbscan_cluster', palette='tab10', s=80)
    plt.title('DBSCAN Clusters (on coordinates)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(title='DBSCAN Cluster')
    plt.show()
    
    # === 8. Cross-tabulate DBSCAN cluster vs Frequency group ===
    ctab = pd.crosstab(df['dbscan_cluster'], df['freq_group_label'])
    print("\nDBSCAN Cluster vs Frequency Group Cross-tabulation:")
    print(ctab)
    
    
    
    """Compare Map from Literatures"""
    # 1. Load image
    image_path = r'C:\Users\McGinley3\Documents\GitHub\lsfm\sample_map\kato.jpg'
    image = imread(image_path)
    
    # 2. Display image and allow user to click 4 points
    %matplotlib tk
    plt.imshow(image)
    plt.title('Click 4 datum points (in order)')
    coords = plt.ginput(4, timeout=0)  # Wait until 4 clicks
    plt.close()
    %matplotlib inline
    
    # 3. Extract RGB and coordinates of all pixels
    height, width, _ = image.shape

    color_codes = []
    points = []
    for y in range(height):
        for x in range(width):
            rgb = image[y, x]  # (R,G,B)
            rgb_tuple = tuple(rgb[:3])  # ensure it's a tuple
    
            coordinate = (x, y)  # (row, col) == (y, x)
            color_codes.append(rgb_tuple)
            points.append(coordinate)
            
    new_coords, new_points = transform_img(coords, points)
    
    
    # === 4. Plot coordinates with freq group ===
    overlay_colors_norm = np.array(color_codes) / 255.0
    custom_palette = {
    'low': 'royalblue',
    'medium': 'gold',
    'high': 'red'
    }
    plt.figure(figsize=(8,6))
    
    plt.title('Coordinates colored by Frequency Group (percentile)')
    plt.xlabel('X')
    plt.ylabel('Y')
    overlay_x, overlay_y = zip(*new_points)
    new_x, new_y = zip(*new_coords)
    plt.scatter(overlay_x, overlay_y, c=overlay_colors_norm, s=80, marker='o', edgecolor='none', label='Overlay points', alpha=0.3)
    sns.scatterplot(data=df, x='x_jitter', y='y_jitter', hue='freq_group_label', palette=custom_palette, s=80)
    plt.scatter(new_x, new_y, c='r')
    plt.legend(title='Frequency Group')
    plt.savefig('tonotopy_TP035.png', dpi=500, bbox_inches='tight')
    plt.show()
