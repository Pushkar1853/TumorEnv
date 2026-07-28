import os
import cv2
import json
import torch
import base64
import zipfile
import requests
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from skimage.color import rgb2gray
from skimage import io, img_as_ubyte
from skimage.util import view_as_blocks

"""### Datasets"""

def extract_zip(zip_path, extract_path):
    # Extract the contents of a zip file to a specified directory
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

"""### DeepLIIF Segmentation """

def infer_image_and_save(images_dir, output_dir, filename):
    # Perform inference using the DeepLIIF API
    def post_image_for_inference(api_url, image_path):
        files = {'img': open(image_path, 'rb')}
        params = {'resolution': '20x'}
        response = requests.post(url=api_url, files=files, params=params)
        return response.json()
    # Convert base64-encoded image to PIL Image
    def b64_to_pil(encoded_image):
        return Image.open(BytesIO(base64.b64decode(encoded_image.encode())))
    # Main logic
    api_url = 'https://deepliif.org/api/infer'
    response_data = post_image_for_inference(api_url, f'{images_dir}/{filename}')
    # Process and save images
    for name, img_base64 in response_data['images'].items():
        output_filepath = f'{output_dir}/{os.path.splitext(filename)[0]}_{name}.png'
        with open(output_filepath, 'wb') as f:
            b64_to_pil(img_base64).save(f, format='PNG')
    # Display scoring information
    scoring_info = json.dumps(response_data['scoring'], indent=2)
    print(scoring_info)


"""#### Preprocessing """

def process_and_save_density_images(input_path, output_dir, colors, target_size=(100, 100)):
    # Load image
    image = io.imread(input_path)
    # Resize image
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    # Convert to grayscale
    resized_image_gray = rgb2gray(resized_image)
    # Initialize labeled masks
    labeled_tumor = np.zeros_like(resized_image_gray, dtype=bool)
    labeled_immune = np.zeros_like(resized_image_gray, dtype=bool)
    labeled_normal = np.zeros_like(resized_image_gray, dtype=bool)
    # Extract regions based on color ranges
    for channel in range(3):
        labeled_tumor = np.logical_or(
            labeled_tumor, np.logical_and(
                resized_image[..., channel] >= colors['tumor'][0][channel],
                resized_image[..., channel] <= colors['tumor'][1][channel]
            )
        )
        labeled_immune = np.logical_or(
            labeled_immune, np.logical_and(
                resized_image[..., channel] >= colors['immune'][0][channel],
                resized_image[..., channel] <= colors['immune'][1][channel]
            )
        )
        labeled_normal = np.logical_or(
            labeled_normal, np.logical_and(
                resized_image[..., channel] >= colors['normal'][0][channel],
                resized_image[..., channel] <= colors['normal'][1][channel]
            )
        )
    # Convert to density images
    tumor_density = img_as_ubyte(labeled_tumor)
    immune_density = img_as_ubyte(labeled_immune)
    normal_density = img_as_ubyte(labeled_normal)
    # Threshold density images
    tumor_density = tumor_density // 255
    immune_density = immune_density // 255
    normal_density = normal_density // 255
    # Save density images
    io.imsave(f'{output_dir}/resized_immune.png', immune_density)
    io.imsave(f'{output_dir}/resized_tumor.png', tumor_density)
    io.imsave(f'{output_dir}/resized_normal.png', normal_density)

"""### For Tumor cells"""

def process_and_save_tumor_data(input_path, output_dir, block_shape=(100, 100)):
    # Load tumor image
    tumor_img = io.imread(input_path)
    # View image as blocks and reshape
    tumor_img_blocks = view_as_blocks(tumor_img, block_shape=block_shape)
    tumor_img_flat = tumor_img_blocks.reshape(-1, *block_shape).ravel()
    # Save tumor density as CSV
    tumor_density_path = f'{output_dir}/Tum_dense.csv'
    np.savetxt(tumor_density_path, tumor_img_flat, delimiter=',', fmt='%d', header='Tum_dense', comments='')
    # Read CSV and create proliferating cells condition
    tumor_data = pd.read_csv(tumor_density_path)
    proliferating_cells_condition = (tumor_data["Tum_dense"] > 0)
    # Generate random values for proliferating cells
    tumor_img_prolif = np.zeros_like(tumor_data["Tum_dense"])
    tumor_img_prolif[proliferating_cells_condition] = np.random.randint(0, 10, np.sum(proliferating_cells_condition))
    # Threshold proliferating cells
    tumor_img_prolif = np.where(tumor_img_prolif > 0, 1, 0)
    # Save proliferating cells as CSV
    tumor_prolif_path = f'{output_dir}/Tum_prolif.csv'
    np.savetxt(tumor_prolif_path, tumor_img_prolif, delimiter=',', fmt='%d', header='Tum_prolif', comments='')
    # Reshape and save tumor IDs as CSV
    tumor_img_blocks_df = pd.DataFrame(tumor_img_blocks.ravel())
    tumor_img_blocks_df = tumor_img_blocks_df[tumor_img_blocks_df[0] >= 0]
    tumor_img_blocks_df = tumor_img_blocks_df.reset_index(drop=True)
    tumor_img_blocks_df = tumor_img_blocks_df.reset_index()
    tumor_img_blocks_df = tumor_img_blocks_df.rename(columns={'index': 'ID'})
    tumor_id_path = f'{output_dir}/tumor_id.csv'
    tumor_img_blocks_df.to_csv(tumor_id_path, index=False, header='id')

"""### For Immune (CD8) cells"""

def process_and_save_immune_data(input_path, output_dir, block_shape=(100, 100)):
    # Load immune image
    immune_img = io.imread(input_path)
    # View image as blocks
    immune_img_blocks = view_as_blocks(immune_img, block_shape=block_shape)
    # Reshape and save immune IDs as CSV
    immune_img_blocks_df = pd.DataFrame(immune_img_blocks.ravel())
    immune_img_blocks_df = immune_img_blocks_df[immune_img_blocks_df[0] >= 0]
    immune_img_blocks_df = immune_img_blocks_df.reset_index(drop=True)
    immune_img_blocks_df = immune_img_blocks_df.reset_index()
    immune_img_blocks_df = immune_img_blocks_df.rename(columns={'index': 'ID'})
    immune_id_path = f'{output_dir}/immune_id.csv'
    immune_img_blocks_df.to_csv(immune_id_path, index=False, header='id')
    # Flatten and save CD8 dense as CSV
    immune_img_dense = immune_img_blocks.ravel()
    immune_img_dense = immune_img_dense.astype(np.uint8)
    cd8_dense_path = f'{output_dir}/CD8_dense.csv'
    np.savetxt(cd8_dense_path, immune_img_dense, delimiter=',', fmt='%d', header='CD8_dense', comments='')
    # Read CD8 dense CSV and create proliferating cells condition
    immune_data = pd.read_csv(cd8_dense_path)
    proliferating_cells_condition = (immune_data["CD8_dense"] > 0)
    # Generate random values for proliferating cells and threshold
    immune_img_prolif = np.zeros_like(immune_data["CD8_dense"])
    immune_img_prolif[proliferating_cells_condition] = np.random.randint(0, 8, np.sum(proliferating_cells_condition))
    immune_img_prolif = np.where(immune_img_prolif > 0, 1, 0)
    # Save proliferating cells as CSV
    cd8_prolif_path = f'{output_dir}/CD8_prolif.csv'
    np.savetxt(cd8_prolif_path, immune_img_prolif, delimiter=',', fmt='%d', header='CD8_prolif', comments='')
    # Read CD8 proliferating cells CSV and create non-proliferating cells condition
    immune_data_2 = pd.read_csv(cd8_prolif_path)
    non_prolif_condition = (immune_data_2["CD8_prolif"] == 0) & proliferating_cells_condition
    # Generate binary mask for non-proliferating cells
    immune_img_non_prolif = np.where(non_prolif_condition, 1, 0)
    # Save non-proliferating cells as CSV
    cd8_non_prolif_path = f'{output_dir}/CD8_non_prolif.csv'
    np.savetxt(cd8_non_prolif_path, immune_img_non_prolif, delimiter=',', fmt='%d', header='CD8_non_prolif', comments='')

""" ### Compute RDF """

def radial_distribution_function(image, radii, tile_size=100, min_counts=10, max_counts=10000):
    # Extract tiles 
    tiles = image.unfold(-2, tile_size, tile_size).unfold(-1, tile_size, tile_size)
    # Get valid tiles
    counts = tiles.sum((-3, -2, -1))
    valid = (counts > min_counts) & (counts < max_counts)
    # Expand valid 
    valid_exp = valid.unsqueeze(2)
    # Index tiles
    tiles = tiles[valid_exp.expand(tiles.shape)]
    # Get coords of all CD8 T-cells
    nonzero_coords = torch.nonzero(tiles > 0, as_tuple=True)
    cd8_coords = torch.stack(nonzero_coords, dim=-1).float()
    # Compute pairwise distances
    dist = torch.cdist(cd8_coords, cd8_coords)  
    # Bin counts hist
    hist = torch.histc(dist, bins=len(radii)-1, min=0, max=radii[-1].item()).float()  
    # Normalize   
    areas = 3.14159 * ((radii[1:]**2) - (radii[:-1]**2))
    reshaped_tiles = tiles.view(tiles.shape[0], -1)
    tile_counts = reshaped_tiles.sum(dim=-2).float()[:, None]
    expected =  areas[:, None] * tile_counts * (tile_counts - 1)
    expected = torch.clamp(expected, min=1e-7)
    rdf = hist / expected
    # Average        
    return rdf.mean(0)

""" ### Compute SAM """

def computeSAM(rdf_sim, rdf_obs, obs_range_tol=0.2, frac_within_tol=0.7):
    num_dists = rdf_obs.shape[0]
    # Calculate acceptable range
    rdf_range = (rdf_obs.max() - rdf_obs.min())
    max_obs = torch.max(rdf_obs + obs_range_tol * rdf_range, torch.zeros_like(rdf_obs))
    min_obs = torch.max(torch.min(rdf_obs - obs_range_tol * rdf_range, torch.zeros_like(rdf_obs)), torch.zeros_like(rdf_obs))
    # Check if simulated RDF is within tolerance range
    within_tol = (rdf_sim > min_obs) & (rdf_sim < max_obs)
    # Calculate fraction of dists where sim RDF is within tol 
    frac_within = torch.sum(within_tol) / num_dists
    # SAM is fraction of dists above threshold
    sam = (frac_within >= frac_within_tol).float()
    sam = torch.autograd.Variable(sam, requires_grad=True)
    return sam, frac_within

""" ### Compute VarSAM """

def computeVarSAM(rdf_sim, rdf_obs):
    # Extract first 15 RDF distances 
    rdf_sim_near = rdf_sim[:15]  
    rdf_obs_near = rdf_obs[:15]
    # Calculate ranges
    range_sim = rdf_sim_near.max() - rdf_sim_near.min()
    range_obs = rdf_obs_near.max() - rdf_obs_near.min()
    # VarSAM is ratio of ranges  
    # Clamp between 0 and 1
    var_sam = torch.min(range_obs / range_sim, torch.tensor(1.0)) 
    var_sam = torch.max(var_sam, torch.tensor(0.0))
    return var_sam


""" ### main """

if __name__ == "__main__":

    images_dir = '/kaggle/input/ihc-cd8-img-2/Auploader/'
    output_dir = '/kaggle/working/'
    filename = '4_Lenti-HPV-07_CD8.tif'

    infer_image_and_save(images_dir, output_dir, filename)

    colors = {'immune': [[200, 0, 0], [255, 13, 13]],
            'tumor': [[0, 0, 225], [13, 13, 255]],
            'normal': [[120, 120, 120], [135, 135, 135]]}

    input_image_path = '/kaggle/working/4_Lenti-HPV-07_CD8_SegOverlaid.png'
    output_directory = '/kaggle/working/'

    process_and_save_density_images(input_image_path, output_directory, colors)

    input_tumor_path = '/kaggle/working/resized_tumor.png'
    output_directory_tumor = '/kaggle/working/'

    process_and_save_tumor_data(input_tumor_path, output_directory_tumor)

    input_immune_path = '/kaggle/working/resized_immune.png'
    output_directory_immune = '/kaggle/working/'

    process_and_save_immune_data(input_immune_path, output_directory_immune)

    immune_density = io.imread('/kaggle/working/resized_immune.png')
    immune_density_tensor = torch.tensor(immune_density, dtype=torch.float32, requires_grad = True).unsqueeze(0).unsqueeze(0)
    radii = torch.tensor(torch.arange(0, 10, dtype=torch.float32), requires_grad=True)

    rdf_out = radial_distribution_function(immune_density_tensor, radii)

    sam, frac_within = computeSAM(rdf_out, rdf_out, obs_range_tol=0.2, frac_within_tol=0.6)
    print(f"SAM: {sam}") 
    print(f"Fraction Within Tolerance: {frac_within:.3f}")

    var_sam = computeVarSAM(rdf_out, rdf_out)
    print(var_sam)

