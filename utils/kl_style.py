import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, findfont, FontProperties

def configure_pandas():
    """Configure pandas display settings."""
    options = {
        'display.max_columns': None,
        'display.max_rows': None,
        'display.width': None,
        'display.max_seq_items': None,
        'display.float_format': '{:.3f}'.format
    }
    for option, value in options.items():
        pd.set_option(option, value)

def set_plotting_style(style='darkgrid', dpi=None):
    """Configure the plotting style for matplotlib/seaborn.
    
    Args:
        style (str): The seaborn style to use ('darkgrid' by default)
        dpi (int): The DPI to use for plotting (None uses default)
    """
    
    # Set the style and palette
    sns.set_style(style)
    sns.set_palette("deep")
    
    # Check if Times font is available
    available_fonts = [f.name for f in fontManager.ttflist]
    if 'Times' in available_fonts:
        plt.rcParams['font.family'] = ['Times']
    # If Times is not available, let matplotlib use its default
    
    # Set DPI if specified
    if dpi is not None:
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
        

def print_font_info(show_all_fonts=False):
    """Print current font settings and actual font being used."""
    # Check configured font
    print("Configured font family:", plt.rcParams['font.family'])
    
    # Check actual font file being used
    font_prop = FontProperties(family=plt.rcParams['font.family'])
    actual_font = findfont(font_prop)
    print("\nActual font file being used:", actual_font)
    
    # Print seaborn style settings
    print("\nSeaborn style settings:")
    print(sns.axes_style())
    
    if show_all_fonts:
        print("\nAvailable fonts on your system:")
        for font in sorted(set([f.name for f in fontManager.ttflist])):
            print(f"- {font}")

def setup(high_quality=False, show_fonts=False):
    """Main setup function to configure all styling preferences."""
    configure_pandas()
    set_plotting_style(dpi=300 if high_quality else None)
    print_font_info(show_all_fonts=show_fonts)
  

# Only run setup if the module is run directly (not imported)
if __name__ == "__main__":
    setup()