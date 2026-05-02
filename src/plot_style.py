import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def configure_journal_style():
    """
    Configures matplotlib for top journal style plotting.
    This function sets global rcParams, defines a log formatter, and provides a color palette.
    Call this function before creating plots to apply the style.
    """
    # -------------------------- 1. Top Journal Style Global Configuration (General, No Need to Modify) --------------------------
    # Recommended to place at the beginning of all plotting code
    plt.rcParams.update(
        {
            # —— Output and Font Embedding (Vector Priority, Editable in Illustrator/AI) ——
            "savefig.format": "pdf",  # Nature recommends submitting PDF (vector)
            "figure.dpi": 300,  # Preview resolution
            "savefig.dpi": 600,  # Clearer raster elements (if any)
            "pdf.fonttype": 3,  # TrueType, convenient for later editing
            "ps.fonttype": 3,
            "svg.fonttype": "path",  # SVG retains text, no curve conversion
            # —— Layout and Size
            # (Single column single figure (3,2), double column double figure (5.5,4), double column triple figure (7.5,1.5), adjust slightly based on width and height) ——
            "figure.figsize": (3, 2),  # Default single column size
            "figure.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.transparent": False,  # Journals usually require white background
            "figure.constrained_layout.use": False,
            # —— Font (Sans-serif, commonly Arial/Helvetica) ——
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
            ],  # Add DejaVu as fallback
            "axes.unicode_minus": True,  # Use real minus sign "−"
            "text.usetex": False,  # Do not enable LaTeX unless necessary
            # —— Math Font and Scientific Notation (Consistent with Sans-serif Body) ——
            "mathtext.fontset": "dejavusans",
            "axes.formatter.use_mathtext": True,
            "axes.formatter.limits": (
                -3,
                3,
            ),  # Switch to scientific notation only when <1e-3 or >1e3
            # —— Font Size (Print 6.5–7 pt common; consistent, compact) ——
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "legend.title_fontsize": 7,
            # —— Axes and Ticks (Four Borders + Inward Ticks + Enable Minor Ticks) ——
            "axes.spines.top": True,  # Set to False to remove the top line of the frame
            "axes.spines.right": True,  # Set to False to remove the right line of the frame
            "axes.linewidth": 0.6,  # Thin border (~0.5–0.7 pt)
            "axes.labelpad": 2.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,  # Show x-axis minor tick lines
            "ytick.minor.visible": True,  # Show y-axis minor tick lines
            "xtick.major.size": 4,
            "xtick.major.width": 0.6,
            "xtick.minor.size": 2,
            "xtick.minor.width": 0.4,
            "xtick.major.pad": 2,
            "ytick.major.size": 4,
            "ytick.major.width": 0.6,
            "ytick.minor.size": 2,
            "ytick.minor.width": 0.4,
            "ytick.major.pad": 2,
            # —— Lines, Error Bars, and Legends ——
            "lines.linewidth": 0.8,  # Thin lines, avoid too thick
            "lines.markersize": 3.0,
            "errorbar.capsize": 2.0,
            "legend.frameon": False,
            "legend.handlelength": 2.0,
            "legend.handleheight": 0.7,
            "legend.borderaxespad": 1,
            "legend.columnspacing": 2,
        }
    )

    def log_format(x, pos):
        """
        Formatter for log axes: simplifies 10^x labels to x.
        """
        if x > 0:
            return f"{int(np.log10(x))}"  # Simplify 10^x label to x
        else:
            return f"{x}"

    # Top Journal Color Palette (General)
    colors = {
        "blue": "#1f77b4",
        "orange": "#ff7f0e",
        "green": "#2ca02c",
        "red": "#d62728",
        "purple": "#9467bd",
        "dark_gray": "#595959",
    }

    return log_format, colors