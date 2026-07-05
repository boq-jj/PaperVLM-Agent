"""Create small ChartQA-style sample chart images for local evaluation."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from _bootstrap import configure_stdio

configure_stdio()


WIDTH = 900
HEIGHT = 620
MARGIN_LEFT = 110
MARGIN_RIGHT = 70
MARGIN_TOP = 95
MARGIN_BOTTOM = 105
BACKGROUND = "white"
AXIS_COLOR = "#222222"
GRID_COLOR = "#d9dee7"
BAR_COLOR = "#4f7db8"
BAR_ALT_COLOR = "#6aa36f"
LINE_COLORS = ["#3d6fb6", "#c75f5f", "#6a9f59"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate local ChartQA-style PNG images used by chartqa_sample.jsonl."
    )
    parser.add_argument(
        "--output-dir",
        default="data/datasets/chartqa/images",
        help="Directory where sample chart images will be written.",
    )
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a readable font with a safe fallback."""
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str = AXIS_COLOR,
) -> None:
    """Draw text centered at the given coordinate."""
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=font, fill=fill)


def value_to_y(value: float, max_value: float) -> float:
    """Map a chart value to canvas y-coordinate."""
    plot_height = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    return HEIGHT - MARGIN_BOTTOM - (value / max_value) * plot_height


def draw_axes(
    draw: ImageDraw.ImageDraw,
    title: str,
    y_label: str,
    max_value: int,
) -> None:
    """Draw chart title, axes, grid lines, and y-axis ticks."""
    title_font = load_font(28, bold=True)
    label_font = load_font(18)
    tick_font = load_font(16)

    draw_centered_text(draw, (WIDTH / 2, 40), title, title_font)
    draw.text((20, MARGIN_TOP - 35), y_label, font=label_font, fill=AXIS_COLOR)

    x0 = MARGIN_LEFT
    y0 = HEIGHT - MARGIN_BOTTOM
    x1 = WIDTH - MARGIN_RIGHT
    y1 = MARGIN_TOP
    draw.line((x0, y0, x1, y0), fill=AXIS_COLOR, width=3)
    draw.line((x0, y0, x0, y1), fill=AXIS_COLOR, width=3)

    step = max_value // 5
    for tick in range(0, max_value + 1, step):
        y = value_to_y(tick, max_value)
        draw.line((x0 - 8, y, x0, y), fill=AXIS_COLOR, width=2)
        draw.line((x0, y, x1, y), fill=GRID_COLOR, width=1)
        draw.text((x0 - 55, y - 10), str(tick), font=tick_font, fill=AXIS_COLOR)


def draw_bar_chart(
    output_path: Path,
    title: str,
    y_label: str,
    labels: list[str],
    values: list[int],
    max_value: int,
    bar_color: str = BAR_COLOR,
) -> None:
    """Draw a simple bar chart image."""
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw_axes(draw, title, y_label, max_value)

    label_font = load_font(18)
    value_font = load_font(18, bold=True)
    plot_width = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    slot_width = plot_width / len(values)
    bar_width = slot_width * 0.52
    baseline = HEIGHT - MARGIN_BOTTOM

    for index, (label, value) in enumerate(zip(labels, values)):
        center_x = MARGIN_LEFT + slot_width * (index + 0.5)
        left = center_x - bar_width / 2
        right = center_x + bar_width / 2
        top = value_to_y(value, max_value)
        color = bar_color if index % 2 == 0 else BAR_ALT_COLOR
        draw.rectangle((left, top, right, baseline), fill=color, outline=AXIS_COLOR)
        draw_centered_text(draw, (center_x, top - 18), str(value), value_font)
        draw_centered_text(draw, (center_x, baseline + 28), label, label_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def draw_line_chart(
    output_path: Path,
    title: str,
    y_label: str,
    x_labels: list[str],
    series: dict[str, list[int]],
    max_value: int,
) -> None:
    """Draw a simple multi-series line chart image."""
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw_axes(draw, title, y_label, max_value)

    label_font = load_font(18)
    value_font = load_font(16, bold=True)
    legend_font = load_font(17)
    plot_width = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    baseline = HEIGHT - MARGIN_BOTTOM
    points_x = [
        MARGIN_LEFT + (plot_width * index / (len(x_labels) - 1))
        for index in range(len(x_labels))
    ]

    for index, label in enumerate(x_labels):
        draw_centered_text(draw, (points_x[index], baseline + 28), label, label_font)

    for series_index, (name, values) in enumerate(series.items()):
        color = LINE_COLORS[series_index % len(LINE_COLORS)]
        points = [(points_x[index], value_to_y(value, max_value)) for index, value in enumerate(values)]
        draw.line(points, fill=color, width=4)
        for point, value in zip(points, values):
            x, y = point
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline=AXIS_COLOR, width=2)
            draw_centered_text(draw, (x, y - 23), str(value), value_font, fill=color)

        legend_x = MARGIN_LEFT + series_index * 160
        legend_y = HEIGHT - 55
        draw.rectangle((legend_x, legend_y, legend_x + 22, legend_y + 14), fill=color)
        draw.text((legend_x + 30, legend_y - 4), name, font=legend_font, fill=AXIS_COLOR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def create_sample_images(output_dir: Path) -> list[Path]:
    """Create all sample chart images and return their paths."""
    outputs = [
        output_dir / "example_001.png",
        output_dir / "example_002.png",
        output_dir / "example_003.png",
        output_dir / "example_004.png",
        output_dir / "example_005.png",
    ]

    draw_bar_chart(
        outputs[0],
        title="Accuracy by Method",
        y_label="Accuracy (%)",
        labels=["Method A", "Method B", "Method C", "Method D"],
        values=[82, 91, 87, 78],
        max_value=100,
    )
    draw_line_chart(
        outputs[1],
        title="Annual Score by Year",
        y_label="Score",
        x_labels=["2018", "2019", "2020", "2021"],
        series={"Value": [35, 42, 49, 53]},
        max_value=60,
    )
    draw_bar_chart(
        outputs[2],
        title="Model Comparison",
        y_label="Score",
        labels=["Model A", "Model B", "Model C"],
        values=[76, 83, 68],
        max_value=100,
    )
    draw_bar_chart(
        outputs[3],
        title="Category Counts",
        y_label="Count",
        labels=["Group A", "Group B", "Group C", "Group D"],
        values=[15, 24, 30, 18],
        max_value=40,
    )
    draw_line_chart(
        outputs[4],
        title="Category Trend from 2020 to 2021",
        y_label="Value",
        x_labels=["2020", "2021"],
        series={
            "Category A": [40, 48],
            "Category B": [35, 39],
            "Category C": [58, 44],
        },
        max_value=70,
    )

    return outputs


def main() -> None:
    """Generate local ChartQA-style sample images."""
    args = parse_args()
    output_paths = create_sample_images(Path(args.output_dir))
    print("Generated ChartQA-style sample images:")
    for path in output_paths:
        print(f"  {path.as_posix()}")


if __name__ == "__main__":
    main()
