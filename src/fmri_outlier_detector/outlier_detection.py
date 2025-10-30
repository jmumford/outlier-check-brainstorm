import gc
import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import get_cmap
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from nilearn import datasets, plotting
from nilearn.image import load_img

# import img2pdf
from PIL import Image


def get_symmetric_percentile_bounds(
    nifti_paths: List[str], percentile: float = 98
) -> float:
    """
    Calculate symmetric percentile bounds that should work well for a single
    colorbar across multiple images

    Args:
        nifti_paths (List[str]): List of paths to NIfTI image files (contrasts/t/zstats).
        percentile (float): Percentile to use for calculating bounds (default: 98).

    Returns:
        float: The symmetric upper bound.

    Raises:
        ValueError: If no valid data is found in the NIfTI images.
    """
    all_data = [load_img(p).get_fdata() for p in nifti_paths]
    all_data_flat = np.concatenate([d.ravel() for d in all_data])
    all_data_flat = all_data_flat[np.isfinite(all_data_flat)]  # exclude NaNs/Infs

    if len(all_data_flat) == 0:
        raise ValueError('No valid data found in the NIfTI images.')

    high = np.percentile(np.abs(all_data_flat), percentile)
    return high


def get_outlier_voxel_percentages(
    nifti_paths: List[str], n_std: float = 2
) -> List[float]:
    """
    Calculate the percentage of outlier voxels for each subject.
    Based on the voxelwise mean and sd across subjects
    Outlier is based on mean +/- n_std * sd

    Args:
        nifti_paths (List[str]): List of paths to image files (contrast images/t/zstats).
        n_std (float): Number of standard deviations to use for outlier detection (default: 2).

    Returns:
        List[float]: List of outlier percentages for each subject.
    """
    data = np.array([load_img(p).get_fdata() for p in nifti_paths])
    voxelwise_mean = np.mean(data, axis=0)
    voxelwise_std = np.std(data, axis=0)

    lower_bound = voxelwise_mean - n_std * voxelwise_std
    upper_bound = voxelwise_mean + n_std * voxelwise_std

    epsilon = 1e-6
    valid_mask = (np.isfinite(voxelwise_std)) & (voxelwise_std > epsilon)

    outlier_percentages = []
    for subject_data in data:
        mask = np.isfinite(subject_data) & valid_mask
        outliers = (subject_data < lower_bound) | (subject_data > upper_bound)
        valid_voxels = np.sum(mask)
        outlier_voxels = np.sum(outliers & mask)
        outlier_percentages.append(100 * outlier_voxels / valid_voxels)

    return outlier_percentages


def summarize_outlier_percentages(
    df_list: List[pd.DataFrame], output_dir: str, temp_dir: Optional[str] = None
) -> Tuple[str, str]:
    """
    Combines a list of DataFrames with subject outlier data, generates summary plots,
    and saves combined data to CSV.

    Args:
        df_list (List[pd.DataFrame]): List of DataFrames with at least
                                      'image_outlier_percentage' and 'contrast_name'
                                      columns.
        output_dir (str): Directory to save plots and CSV.
        temp_dir (Optional[str]): Temporary directory for intermediate files.
                                  If None, uses output_dir.

    Returns:
        Tuple[str, str]: Paths to the histogram image files.
            (1) Combined histogram
            (2) Faceted histogram by contrast
    """
    if temp_dir is None:
        temp_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    combined_df = pd.concat(df_list, ignore_index=True)

    path_all = _plot_combined_histogram(combined_df, temp_dir)
    path_by_contrast = _plot_faceted_histogram(combined_df, temp_dir)

    csv_path = os.path.join(output_dir, 'percent_outlier_data.csv')
    combined_df.to_csv(csv_path, index=False)

    return [path_all, path_by_contrast]


def _plot_combined_histogram(df: pd.DataFrame, output_dir: str) -> str:
    """
    Plot and save a combined histogram of outlier percentages.

    Args:
        df (pd.DataFrame): DataFrame containing outlier percentages.
        output_dir (str): Directory to save the plot.

    Returns:
        str: Path to the saved histogram image.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        df['image_outlier_percentage'],
        bins=30,
        kde=False,
        ax=ax,
    )
    ax.set_title('Distribution of Outlier Percentages\n(All Input Files)')
    ax.set_xlabel('Input Image Outlier Percentage')
    ax.set_ylabel('Frequency')

    path_all = os.path.join(output_dir, 'outlier_percentage_dist_all.png')
    fig.tight_layout()
    fig.savefig(path_all, dpi=300)
    plt.close(fig)
    return path_all


def _plot_faceted_histogram(df: pd.DataFrame, output_dir: str) -> str:
    """
    Plot and save a faceted histogram of outlier percentages by contrast.

    Args:
        df (pd.DataFrame): DataFrame containing outlier percentages and contrast names.
        output_dir (str): Directory to save the plot.

    Returns:
        str: Path to the saved faceted histogram image.
    """
    g = sns.displot(
        df,
        x='image_outlier_percentage',
        col='contrast_name',
        col_wrap=5,
        bins=20,
        facet_kws={'sharex': False, 'sharey': False},
        height=3,
        aspect=1.2,
    )
    g.set_titles('{col_name}')
    g.set_axis_labels('Image-Specific Outlier Percentage', 'Frequency')

    path_by_contrast = os.path.join(output_dir, 'outlier_percentage_dist_by_image.png')
    plt.tight_layout()
    g.savefig(path_by_contrast, dpi=300)
    plt.close(g.fig)
    return path_by_contrast


def _plot_subject_grid(
    subject_labels: List[str],
    nifti_paths: List[str],
    outlier_percentages: List[float],
    mni_mask: np.ndarray,
    contrast_name: str,
    vmax: float,
    vmin: float,
    colorbar_title: Optional[str],
    n_std: float,
) -> plt.Figure:
    """
    Plot a grid of subject images with outlier percentages.

    Args:
        subject_labels (List[str]): Labels for each subject.
        nifti_paths (List[str]): Paths to image files (contrasts/t/zstats).
        outlier_percentages (List[float]): Outlier percentages for each subject.
        mni_mask (np.ndarray): MNI mask for adding contours.
        contrast_name (str): Name of the contrast for the title.
        vmax (float): Maximum value for the colorbar.
        vmin (float): Minimum value for the colorbar.
        colorbar_title (Optional[str]): Title for the colorbar.
        n_std (float): Number of standard deviations used for outlier detection.

    Returns:
        plt.Figure: The created figure object.
    """
    n_subjects = len(subject_labels)
    ncols = (
        4
        if n_subjects <= 12
        else 5
        if n_subjects <= 30
        else 6
        if n_subjects <= 60
        else 10
    )
    nrows = int(np.ceil(n_subjects / ncols))

    subplot_width, subplot_height = 2.0, 1.6
    fig_width = ncols * subplot_width
    fig_height = nrows * subplot_height + 1.5
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(nrows, ncols, figure=fig, wspace=0.1, hspace=0.25)

    title_fontsize = 9 if n_subjects <= 20 else 7 if n_subjects <= 50 else 5

    for i, (label, path, outlier) in enumerate(
        zip(subject_labels, nifti_paths, outlier_percentages)
    ):
        print(f'Processing contrast {i + 1}/{n_subjects}: {label}')
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs[row, col])

        display = plotting.plot_stat_map(
            path,
            display_mode='z',
            cut_coords=[5],
            colorbar=False,
            vmax=vmax,
            vmin=vmin,
            title=None,
            axes=ax,
            bg_img=None,
            annotate=False,
        )
        display.add_contours(mni_mask, colors='greenyellow', linewidths=1.5)

        title = f'{label}\n({outlier:.1f}% > {n_std}SD)'
        ax.set_title(title, fontsize=title_fontsize, pad=4)

    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    cmap = get_cmap('cold_hot')
    norm = Normalize(vmin=vmin, vmax=vmax)
    cbar = ColorbarBase(cbar_ax, cmap=cmap, norm=norm)
    cbar.set_label(colorbar_title, fontsize=10)

    fig.suptitle(contrast_name, fontsize=14, y=0.98)
    return fig


def _build_outlier_summary_df(
    subject_labels: List[str],
    image_outlier_percentages: List[float],
    contrast_name: str,
) -> pd.DataFrame:
    """
    Build a summary DataFrame of outlier percentages.

    Args:
        subject_labels (List[str]): Labels for each subject.
        image_outlier_percentages (List[float]): Outlier percentages for each subject.
        contrast_name (str): Name of the contrast.

    Returns:
        pd.DataFrame: Summary DataFrame containing outlier information.
    """
    data = {
        'subject_label': subject_labels,
        'image_outlier_percentage': image_outlier_percentages,
        'contrast_name': [contrast_name] * len(subject_labels),
    }
    return pd.DataFrame(data)


def get_mean_std_bounds(
    nifti_paths: List[str], n_std: float = 2
) -> Tuple[float, float]:
    """
    Calculate mean and standard deviation bounds for a set of NIfTI images.

    Args:
        nifti_paths (List[str]): List of paths to NIfTI image files (contrasts/t/zstats).
        n_std (float): Number of standard deviations to use for the bounds (default: 2).

    Returns:
        Tuple[float, float]: Lower and upper bounds (vmin, vmax).
    """
    all_data = [load_img(p).get_fdata() for p in nifti_paths]
    all_data_flat = np.concatenate([d.ravel() for d in all_data])
    all_data_flat = all_data_flat[np.isfinite(all_data_flat)]  # exclude NaNs/Infs

    mean = np.mean(all_data_flat)
    std = np.std(all_data_flat)

    vmin = mean - n_std * std
    vmax = mean + n_std * std
    return vmin, vmax


# def combine_pngs_to_pdf(png_files: List[str], pdf_path: str) -> None:
#     """
#     Combine PNG files into a single PDF.

#     Args:
#         png_files (List[str]): List of PNG file paths.
#         pdf_path (str): Output PDF file path.
#     """
#     if not png_files:
#         print('No PNG files to combine')
#         return

#     print(f'Combining {len(png_files)} PNG files into PDF: {pdf_path}')

#     os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

#     try:
#         with open(pdf_path, 'wb') as f:
#             f.write(img2pdf.convert(png_files))
#         print(f'PDF created successfully: {pdf_path}')
#     except Exception as e:
#         print(f'Error creating PDF: {e}')


def combine_pngs_to_pdf(png_files: List[str], pdf_path: str) -> None:
    """
    Combine PNG files into a single PDF using Pillow.

    Args:
        png_files (List[str]): List of PNG file paths.
        pdf_path (str): Output PDF file path.
    """
    if not png_files:
        print('No PNG files to combine')
        return

    print(f'Combining {len(png_files)} PNG files into PDF: {pdf_path}')

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    try:
        images = [Image.open(f).convert('RGB') for f in png_files]
        images[0].save(pdf_path, save_all=True, append_images=images[1:])
        print(f'PDF created successfully: {pdf_path}')
    except Exception as e:
        print(f'Error creating PDF: {e}')


def generate_png_files_sd_from_mean(
    image_labels: List[str],
    nifti_paths: List[str],
    output_dir: str,
    main_title: str,
    colorbar_title: Optional[str] = None,
    n_std: float = 2,
) -> Tuple[str, pd.DataFrame]:
    """
    Generate PNG files showing standard deviation from mean for contrast images.

    Args:
        image_labels (List[str]): Labels for each contrast image (typically the subject ID).
        nifti_paths (List[str]): Paths to image files (contrast images/t/zstats).
        output_dir (str): Directory to save output PNG files.
        main_title (str): Main heading for the figure (typically a contrast name).
        colorbar_title (Optional[str]): Title for the colorbar,
                                        typically the image
                                        type (contrast estimate/zstat).
        n_std (float): Number of standard deviations to use for outlier detection
                       (default: 2).

    Returns:
        Tuple[str, pd.DataFrame]: Path to the PNG file and a summary dataframe.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_outlier_percentages = get_outlier_voxel_percentages(nifti_paths, n_std=n_std)
    vmax = get_symmetric_percentile_bounds(nifti_paths)
    vmin = -vmax
    mni_mask = datasets.load_mni152_brain_mask()

    fig = _plot_subject_grid(
        image_labels,
        nifti_paths,
        image_outlier_percentages,
        mni_mask,
        main_title,
        vmax,
        vmin,
        colorbar_title,
        n_std,
    )

    png_path = os.path.join(output_dir, f'{main_title}_slice_grid.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f'Saved {png_path}')
    plt.close(fig)
    del fig
    gc.collect()
    time.sleep(0.5)

    summary_df = _build_outlier_summary_df(
        image_labels, image_outlier_percentages, main_title
    )

    return png_path, summary_df


@dataclass
class DataDictionary:
    main_title: str
    nifti_paths: List[str]
    image_labels: List[str]
    data_type_label: str


def validate_data_dictionary(data_dict: Dict[str, Any]) -> DataDictionary:
    """Validate and convert a dictionary to a DataDictionary object."""
    required_keys = ['main_title', 'nifti_paths', 'image_labels', 'data_type_label']
    for key in required_keys:
        if key not in data_dict:
            raise ValueError(f'Missing required key: {key}')

    if len(data_dict['nifti_paths']) != len(data_dict['image_labels']):
        raise ValueError("Length of 'nifti_paths' and 'image_labels' must be the same")

    return DataDictionary(**{k: data_dict[k] for k in required_keys})


def generate_all_data_summaries(
    data_dictionaries: List[Dict[str, Any]],
    n_std: float = 2,
    output_dir: str = './data_summary_output',
) -> None:
    """
    Generate a multi-page PDF summarizing brain imaging data from multiple contrasts or
    studies, to help in identifying subjects who are outliers.

    This function processes a list of dictionaries where each dictionary describes a
    collection of related data (e.g. a specific contrast).  Each dictionary will have a
    page in the PDF displaying a slice of data for each input image and quantifies what
    percentage of voxels are outliers compared to the rest of the data

    Args:
        data_dictionaries (List[Dict[str, Any]]): A list of dictionaries, each containing:
            - 'main_title' (str): The main title for each grid of images (e.g., contrast name)
            - 'nifti_paths' (List[str]): Paths to the NIfTI files (contrast maps, t-stats, or z-stats)
            - 'image_labels' (List[str]): Labels for each panel of the plot (e.g., subject IDs)
            - 'data_type_label' (str): Description of what the voxel intensities
                                       represent.  Used for the colorbar label

        n_std (float, optional): Number of standard deviations to use for outlier detection.
                                 Defaults to 2.

        output_dir (str, optional): Directory to save the output PDF and temporary files.
                                    Defaults to "./data_summary_output".

    Raises:
        ValueError: If the input data dictionaries are not in the correct format.

    Returns:
        None. The function saves a PDF file named 'outlier_analysis.pdf' and a
        percent_outlier_data.csv file with the outlier percentages and image_labels for
        each set of data (described by main_title) in the specified output directory.

    Notes:
        - The output PDF includes:
            1. A page for each input dictionary showing one slice of each input image
            2. A histogram of percent outliers across all subjects and input data
            3. Histograms of percent outliers corresponding to each input dictionary
        - Temporary files are created and then deleted after the PDF is generated
    """
    temp_output_dir = os.path.join(output_dir, 'temp')

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(temp_output_dir)

    # Validate input data
    validated_data = [validate_data_dictionary(d) for d in data_dictionaries]

    file_names = []
    outlier_data_all = []
    for data in validated_data:
        file_name, outlier_data = generate_png_files_sd_from_mean(
            data.image_labels,
            data.nifti_paths,
            temp_output_dir,
            data.main_title,
            n_std=n_std,
            colorbar_title=data.data_type_label,
        )
        file_names.append(file_name)
        outlier_data_all.append(outlier_data)

    all_img_paths = summarize_outlier_percentages(
        outlier_data_all, output_dir, temp_output_dir
    )
    all_img_paths.extend(file_names)
    combine_pngs_to_pdf(all_img_paths, os.path.join(output_dir, 'outlier_analysis.pdf'))
    shutil.rmtree(temp_output_dir)
