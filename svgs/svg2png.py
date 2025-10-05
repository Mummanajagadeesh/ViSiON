import os
import requests
import cairosvg

# List of SVG URLs with -light.svg
svg_urls = [
    "https://raw.githubusercontent.com/Mummanajagadeesh/Mummanajagadeesh/main/repos/vision-light.svg",
    "https://raw.githubusercontent.com/Mummanajagadeesh/Mummanajagadeesh/main/repos/never-light.svg",
    "https://raw.githubusercontent.com/Mummanajagadeesh/Mummanajagadeesh/main/repos/systolic-array-matrix-multiplication-light.svg",
    "https://raw.githubusercontent.com/Mummanajagadeesh/Mummanajagadeesh/main/repos/hw-multiply-and-accumulate-units-verilog-light.svg",
    "https://raw.githubusercontent.com/Mummanajagadeesh/Mummanajagadeesh/main/repos/cordic-algorithm-verilog-light.svg",
    "https://raw.githubusercontent.com/Mummanajagadeesh/Mummanajagadeesh/main/repos/improve-light.svg"
]

# Create output directories
os.makedirs("svgs", exist_ok=True)
os.makedirs("pngs", exist_ok=True)

# Scale factor for higher resolution (2-5 is usually good)
SCALE = 4  # adjust for sharper images

for url in svg_urls:
    try:
        # Get filename from URL
        filename = url.split("/")[-1]
        svg_path = os.path.join("svgs", filename)
        png_path = os.path.join("pngs", filename.replace("-light.svg", ".png"))

        # Download SVG
        r = requests.get(url)
        if r.status_code == 200:
            with open(svg_path, "wb") as f:
                f.write(r.content)
            print(f"Downloaded {filename}")

            # Convert SVG to high-res PNG
            cairosvg.svg2png(url=svg_path, write_to=png_path, scale=SCALE)
            print(f"Converted {filename} -> {png_path} (high-res)")
        else:
            print(f"Failed to download {filename}: HTTP {r.status_code}")
    except Exception as e:
        print(f"Error processing {url}: {e}")
