#!/usr/bin/env python3
"""
Generate a semi-abstract system diagram for PowerPoint presentations.

Shows all components modeled in the ESTAT energy system:
- Solar PV
- Battery storage
- Grid connection
- Heat pump
- Buffer tank
- Building thermal mass
- Indoor/outdoor environment
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Polygon
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'output' / 'xtra'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette (statistik.bs.ch inspired)
COLORS = {
    'solar_yellow': '#FFB800',
    'solar_orange': '#FF8C00',
    'battery_green': '#2a9749',
    'battery_light': '#7BC67B',
    'grid_blue': '#079bca',
    'grid_dark': '#1e4557',
    'heat_red': '#E85A4F',
    'heat_orange': '#FF6B35',
    'building_warm': '#F5E6D3',
    'building_outline': '#8B7355',
    'outdoor_blue': '#87CEEB',
    'outdoor_cold': '#4A90A4',
    'purple': '#9156b4',
    'text': '#333333',
    'arrow_electric': '#1e4557',
    'arrow_heat': '#E85A4F',
    'white': '#FFFFFF',
}


def draw_sun(ax, x, y, size=0.8):
    """Draw a stylized sun."""
    # Sun body
    sun = Circle((x, y), size * 0.4, color=COLORS['solar_yellow'],
                 ec=COLORS['solar_orange'], linewidth=2, zorder=10)
    ax.add_patch(sun)

    # Sun rays
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        x1 = x + size * 0.5 * np.cos(angle)
        y1 = y + size * 0.5 * np.sin(angle)
        x2 = x + size * 0.7 * np.cos(angle)
        y2 = y + size * 0.7 * np.sin(angle)
        ax.plot([x1, x2], [y1, y2], color=COLORS['solar_orange'],
                linewidth=3, solid_capstyle='round', zorder=9)


def draw_pv_panels(ax, x, y, width=1.5, height=0.8):
    """Draw solar PV panels."""
    # Panel frame
    panel = FancyBboxPatch((x - width/2, y - height/2), width, height,
                           boxstyle="round,pad=0.02,rounding_size=0.05",
                           facecolor='#2C3E50', edgecolor='#1a252f',
                           linewidth=2, zorder=5)
    ax.add_patch(panel)

    # Grid lines
    for i in range(1, 4):
        ax.plot([x - width/2 + i*width/4, x - width/2 + i*width/4],
                [y - height/2, y + height/2], color='#1a252f', linewidth=1, zorder=6)
    for i in range(1, 3):
        ax.plot([x - width/2, x + width/2],
                [y - height/2 + i*height/3, y - height/2 + i*height/3],
                color='#1a252f', linewidth=1, zorder=6)

    # Reflection highlight
    ax.plot([x - width/2 + 0.1, x - width/4], [y + height/2 - 0.1, y],
            color='#5D6D7E', linewidth=2, alpha=0.5, zorder=6)


def draw_battery(ax, x, y, width=0.8, height=1.2):
    """Draw a battery storage unit."""
    # Battery body
    battery = FancyBboxPatch((x - width/2, y - height/2), width, height,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor=COLORS['battery_green'],
                             edgecolor='#1e6b35', linewidth=2, zorder=5)
    ax.add_patch(battery)

    # Battery terminal
    terminal = FancyBboxPatch((x - width/6, y + height/2), width/3, height/8,
                              boxstyle="round,pad=0.01,rounding_size=0.02",
                              facecolor='#555', edgecolor='#333',
                              linewidth=1, zorder=6)
    ax.add_patch(terminal)

    # Charge level indicator
    for i in range(3):
        level_y = y - height/3 + i * height/4
        level = FancyBboxPatch((x - width/3, level_y), width*2/3, height/6,
                               boxstyle="round,pad=0.01,rounding_size=0.02",
                               facecolor=COLORS['battery_light'],
                               edgecolor=COLORS['battery_green'],
                               linewidth=1, zorder=6)
        ax.add_patch(level)


def draw_grid(ax, x, y, size=1.0):
    """Draw a power grid symbol (transmission tower simplified)."""
    # Pole
    ax.plot([x, x], [y - size/2, y + size/2], color=COLORS['grid_dark'],
            linewidth=4, zorder=5)

    # Cross arms
    ax.plot([x - size/3, x + size/3], [y + size/4, y + size/4],
            color=COLORS['grid_dark'], linewidth=3, zorder=5)
    ax.plot([x - size/4, x + size/4], [y, y],
            color=COLORS['grid_dark'], linewidth=3, zorder=5)

    # Wires
    for offset in [-size/3, 0, size/3]:
        ax.plot([x + offset, x + offset + size/4], [y + size/4, y + size/2],
                color=COLORS['grid_blue'], linewidth=2, zorder=4)


def draw_heat_pump(ax, x, y, width=1.2, height=1.0):
    """Draw a heat pump unit."""
    # Main body
    body = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor=COLORS['white'],
                          edgecolor=COLORS['grid_dark'], linewidth=2, zorder=5)
    ax.add_patch(body)

    # Fan grille
    fan_x = x - width/4
    fan = Circle((fan_x, y), height/3, facecolor='#E8E8E8',
                 edgecolor=COLORS['grid_dark'], linewidth=1.5, zorder=6)
    ax.add_patch(fan)

    # Fan blades
    for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
        x1 = fan_x + height/6 * np.cos(angle)
        y1 = y + height/6 * np.sin(angle)
        ax.plot([fan_x, x1], [y, y1], color=COLORS['grid_dark'],
                linewidth=2, zorder=7)

    # Heat indicator
    for i, offset in enumerate([0.15, 0.25, 0.35]):
        wave_x = x + width/6
        ax.plot([wave_x + offset, wave_x + offset],
                [y - height/4 + i*0.15, y - height/4 + i*0.15 + 0.1],
                color=COLORS['heat_red'], linewidth=2, zorder=7)


def draw_buffer_tank(ax, x, y, width=0.7, height=1.4):
    """Draw a thermal buffer tank."""
    # Tank body (cylinder representation)
    tank = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.02,rounding_size=0.15",
                          facecolor='#E8E8E8',
                          edgecolor=COLORS['grid_dark'], linewidth=2, zorder=5)
    ax.add_patch(tank)

    # Temperature gradient (hot top, warm bottom)
    gradient_height = height * 0.7
    for i, frac in enumerate(np.linspace(0, 1, 5)):
        color = plt.cm.YlOrRd(0.3 + 0.5 * (1 - frac))
        rect_y = y - height/3 + frac * gradient_height
        rect = FancyBboxPatch((x - width/3, rect_y), width*2/3, gradient_height/5,
                              boxstyle="round,pad=0.01,rounding_size=0.02",
                              facecolor=color, edgecolor='none', zorder=6)
        ax.add_patch(rect)


def draw_house(ax, x, y, width=4, height=3):
    """Draw a simplified house outline."""
    # Walls
    walls = FancyBboxPatch((x - width/2, y - height/2), width, height,
                           boxstyle="round,pad=0.02,rounding_size=0.1",
                           facecolor=COLORS['building_warm'],
                           edgecolor=COLORS['building_outline'],
                           linewidth=3, zorder=2)
    ax.add_patch(walls)

    # Roof
    roof_points = [
        (x - width/2 - 0.2, y + height/2),
        (x, y + height/2 + 1),
        (x + width/2 + 0.2, y + height/2)
    ]
    roof = Polygon(roof_points, closed=True,
                   facecolor='#8B4513', edgecolor='#5D3A1A',
                   linewidth=3, zorder=3)
    ax.add_patch(roof)

    # Window
    window = FancyBboxPatch((x - width/4, y), width/3, height/3,
                            boxstyle="round,pad=0.02,rounding_size=0.05",
                            facecolor='#87CEEB', edgecolor=COLORS['building_outline'],
                            linewidth=2, zorder=3)
    ax.add_patch(window)

    # Window cross
    ax.plot([x - width/4, x - width/4 + width/3], [y + height/6, y + height/6],
            color=COLORS['building_outline'], linewidth=1.5, zorder=4)
    ax.plot([x - width/4 + width/6, x - width/4 + width/6], [y, y + height/3],
            color=COLORS['building_outline'], linewidth=1.5, zorder=4)


def draw_thermometer(ax, x, y, height=0.6, temp='warm'):
    """Draw a small thermometer symbol."""
    color = COLORS['heat_red'] if temp == 'warm' else COLORS['outdoor_cold']

    # Bulb
    bulb = Circle((x, y - height/3), height/6, facecolor=color,
                  edgecolor=COLORS['grid_dark'], linewidth=1, zorder=10)
    ax.add_patch(bulb)

    # Tube
    tube = FancyBboxPatch((x - height/12, y - height/4), height/6, height/2,
                          boxstyle="round,pad=0.01,rounding_size=0.02",
                          facecolor=COLORS['white'], edgecolor=COLORS['grid_dark'],
                          linewidth=1, zorder=9)
    ax.add_patch(tube)

    # Fill
    fill_height = height/3 if temp == 'warm' else height/6
    fill = FancyBboxPatch((x - height/16, y - height/4), height/8, fill_height,
                          boxstyle="round,pad=0.005,rounding_size=0.01",
                          facecolor=color, edgecolor='none', zorder=10)
    ax.add_patch(fill)


def draw_flow_arrow(ax, start, end, color, label='', curved=False, style='electric'):
    """Draw an arrow representing energy/heat flow."""
    if curved:
        # Curved arrow
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2 + 0.5

        arrow = FancyArrowPatch(start, end,
                               connectionstyle=f"arc3,rad=0.3",
                               arrowstyle='-|>',
                               mutation_scale=15,
                               linewidth=3 if style == 'electric' else 4,
                               color=color,
                               zorder=8)
    else:
        arrow = FancyArrowPatch(start, end,
                               arrowstyle='-|>',
                               mutation_scale=15,
                               linewidth=3 if style == 'electric' else 4,
                               color=color,
                               zorder=8)
    ax.add_patch(arrow)

    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.3, label, ha='center', va='bottom',
                fontsize=9, color=color, fontweight='bold', zorder=11)


def create_system_diagram():
    """Create the complete system diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 9)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(6, 8.5, 'ESTAT Energy System Model',
            ha='center', va='center', fontsize=20, fontweight='bold',
            color=COLORS['grid_dark'])

    # === OUTDOOR ENVIRONMENT (left side) ===
    # Sun
    draw_sun(ax, 1.5, 7, size=1.0)
    ax.text(1.5, 5.8, 'Solar\nIrradiance', ha='center', va='top',
            fontsize=10, color=COLORS['solar_orange'], fontweight='bold')

    # Outdoor temperature
    draw_thermometer(ax, 0.8, 3, height=0.8, temp='cold')
    ax.text(0.8, 2.0, 'T_outdoor', ha='center', va='top',
            fontsize=10, color=COLORS['outdoor_cold'], fontweight='bold')

    # === BUILDING (center) ===
    draw_house(ax, 6, 4, width=5, height=3.5)

    # Indoor temperature indicator
    draw_thermometer(ax, 5.5, 4.5, height=0.6, temp='warm')
    ax.text(5.5, 3.7, 'T_indoor', ha='center', va='top',
            fontsize=9, color=COLORS['heat_red'], fontweight='bold')

    # Thermal mass indicator (stylized block pattern)
    for i in range(3):
        for j in range(2):
            block = FancyBboxPatch((6.2 + i*0.2, 4.0 + j*0.25), 0.18, 0.22,
                                   boxstyle="round,pad=0.01,rounding_size=0.02",
                                   facecolor='#C4A484', edgecolor='#8B7355',
                                   linewidth=1, zorder=10)
            ax.add_patch(block)
    ax.text(6.5, 3.5, 'Thermal\nMass', ha='center', va='top',
            fontsize=8, color=COLORS['building_outline'])

    # === ELECTRICAL COMPONENTS (top) ===
    # PV Panels (on roof)
    draw_pv_panels(ax, 5.2, 6.3, width=1.8, height=0.7)
    ax.text(5.2, 5.4, 'PV Array', ha='center', va='top',
            fontsize=10, color=COLORS['grid_dark'], fontweight='bold')

    # Battery
    draw_battery(ax, 10, 6.5, width=0.9, height=1.3)
    ax.text(10, 5.0, 'Battery\nStorage', ha='center', va='top',
            fontsize=10, color=COLORS['battery_green'], fontweight='bold')

    # Grid
    draw_grid(ax, 12, 6.5, size=1.2)
    ax.text(12, 5.0, 'Grid', ha='center', va='top',
            fontsize=10, color=COLORS['grid_dark'], fontweight='bold')

    # === HEATING COMPONENTS (bottom/right) ===
    # Heat pump
    draw_heat_pump(ax, 10, 2.5, width=1.4, height=1.1)
    ax.text(10, 1.2, 'Heat Pump', ha='center', va='top',
            fontsize=10, color=COLORS['grid_dark'], fontweight='bold')

    # Buffer tank
    draw_buffer_tank(ax, 8, 1.5, width=0.7, height=1.5)
    ax.text(8, -0.3, 'Buffer\nTank', ha='center', va='top',
            fontsize=10, color=COLORS['heat_orange'], fontweight='bold')

    # Heating circuit label (HK2)
    ax.text(6.5, 1.8, 'T_HK2', ha='center', va='center',
            fontsize=9, color=COLORS['heat_red'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                     edgecolor=COLORS['heat_red'], alpha=0.9), zorder=10)

    # === ENERGY FLOWS ===
    # Color definitions for flows
    electric_color = '#000000'  # Black for electricity
    heat_color = COLORS['heat_red']  # Red for heat

    # Solar to PV (electricity generation - black)
    draw_flow_arrow(ax, (2.3, 6.8), (4.2, 6.4), COLORS['solar_yellow'],
                   'Irradiance', style='heat')

    # PV electrical outputs (black)
    draw_flow_arrow(ax, (6.2, 6.3), (9.4, 6.5), electric_color,
                   'Charge', curved=True)
    draw_flow_arrow(ax, (6.2, 6.0), (9.2, 3.0), electric_color,
                   'Direct', curved=True)

    # Battery to consumption (black)
    draw_flow_arrow(ax, (9.5, 5.8), (8.5, 4.0), electric_color,
                   'Discharge', curved=True)

    # Grid connections (black)
    draw_flow_arrow(ax, (11.2, 6.5), (10.6, 6.5), electric_color, 'Import')
    draw_flow_arrow(ax, (10.6, 6.2), (11.2, 6.2), electric_color, 'Export')

    # Heat pump to buffer (red)
    draw_flow_arrow(ax, (9.2, 2.3), (8.4, 2.0), heat_color, '', style='heat')

    # Buffer to building (heating circuit) (red)
    draw_flow_arrow(ax, (7.6, 1.8), (6.9, 2.2), heat_color, '', style='heat')

    # Heat flow into building (red)
    draw_flow_arrow(ax, (6.2, 2.5), (5.8, 3.2), heat_color, '', style='heat')

    # Heat loss to outdoor (red)
    draw_flow_arrow(ax, (3.5, 3.5), (2.0, 3.0), heat_color,
                   'Heat Loss', style='heat')

    # === LEGEND ===
    legend_y = 0.3
    legend_x = 0.5

    # Electrical flow (black)
    ax.plot([legend_x, legend_x + 0.8], [legend_y + 0.6, legend_y + 0.6],
            color='#000000', linewidth=3)
    ax.annotate('', xy=(legend_x + 0.8, legend_y + 0.6),
                xytext=(legend_x + 0.6, legend_y + 0.6),
                arrowprops=dict(arrowstyle='-|>', color='#000000'))
    ax.text(legend_x + 1.0, legend_y + 0.6, 'Electrical flow',
            va='center', fontsize=9, color='#000000')

    # Heat flow (red)
    ax.plot([legend_x, legend_x + 0.8], [legend_y, legend_y],
            color=COLORS['heat_red'], linewidth=4)
    ax.annotate('', xy=(legend_x + 0.8, legend_y),
                xytext=(legend_x + 0.6, legend_y),
                arrowprops=dict(arrowstyle='-|>', color=COLORS['heat_red']))
    ax.text(legend_x + 1.0, legend_y, 'Heat flow',
            va='center', fontsize=9, color=COLORS['heat_red'])

    plt.tight_layout()

    # Save in multiple formats
    output_path_png = OUTPUT_DIR / 'system_diagram.png'
    output_path_svg = OUTPUT_DIR / 'system_diagram.svg'
    output_path_pdf = OUTPUT_DIR / 'system_diagram.pdf'

    fig.savefig(output_path_png, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(output_path_svg, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(output_path_pdf, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"Saved: {output_path_png}")
    print(f"Saved: {output_path_svg}")
    print(f"Saved: {output_path_pdf}")

    plt.close()

    return output_path_png


def main():
    """Generate system diagram."""
    print("=" * 60)
    print("Generating System Diagram")
    print("=" * 60)

    output_path = create_system_diagram()

    print("\nDone!")
    print(f"\nUse {output_path.name} for PowerPoint (300 DPI PNG)")
    print("Use system_diagram.svg for scalable graphics")
    print("Use system_diagram.pdf for print quality")


if __name__ == '__main__':
    main()
