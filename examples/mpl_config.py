"""Shared matplotlib configuration for example notebooks."""

import matplotlib.pyplot as plt


def apply_mpl_config():
    """Apply the shared plotting style used in example notebooks."""
    # Parameter for a figure with one axis only: figures should be assembled in inkscape
    plt.rcParams['font.size'] = 7
    plt.rcParams['axes.labelsize'] = 7
    plt.rcParams['axes.titlesize'] = 7
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = (7.18 / 3, 7.18 / 3)
    plt.rcParams['legend.fontsize'] = 6
    plt.rcParams['lines.markersize'] = 3.0
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.01

    plt.rcParams.update({
        # Use mathtext, not LaTeX
        'text.usetex': False,
        # Use the Computer modern font
        'font.family': 'sans-serif',
        'font.serif': 'cmr10',
        'font.sans-serif': 'Nimbus Sans',
        'axes.formatter.use_mathtext': True,
        'mathtext.fontset': 'cm',
        # Use ASCII minus
        'axes.unicode_minus': False,
    })
