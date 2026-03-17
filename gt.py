"""
Scripts for preparing training samples from Page XML GT
"""

from PIL.Image import Image as ImageT
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm


class Page:
    """
    A ground truth page.
    """

    def __init__(self, xml_path, image_path):
        self.xml_path = xml_path
        self.image_path = image_path
        self.key = xml_path

    def lines(self) -> list[tuple[str, ImageT, str]]:
        """
        Return the page's lines as a list of (line_id, image, text) tuples.
        """
        xml = ET.parse(self.xml_path, ET.XMLParser(encoding="utf-8"))
        image = Image.open(self.image_path).convert("RGB")

        ns = {"": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        lines = xml.findall(".//TextLine", ns)
        results = []
        for i, line in enumerate(lines):
            coords = line.find("Coords", ns)
            if coords is None:
                print(f"{self.key}: line does not have any coordinates!")
                continue
            coords = coords.get("points")
            polygon = coords2polygon(coords)
            content = line.find("./TextEquiv/Unicode", ns)
            if content is None or content.text is None or content.text.isspace():
                continue
            try:
                masked_image = apply_mask(image, polygon)
            except ValueError:
                continue

            line_id = f"{self.key}_line{i}"
            results.append((line_id, masked_image, content.text))
        return results


def polygon2bbox(polygon: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """
    Get bounding box of polygon.
    """
    xs = [x for x, _ in polygon]
    ys = [y for _, y in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def coords2polygon(coords: str) -> list[tuple[int, int]]:
    """
    Convert coordinate string to polygon.
    """
    coords = coords.split()
    polygon = [tuple(map(int, coord.split(","))) for coord in coords]
    return polygon


def apply_mask(image: Image, polygon: list[tuple[int, int]]) -> Image:
    """
    Apply polygon mask to image.
    """
    # Crop image first, it makes cv2.fillPoly a lot quicker.
    bbox = polygon2bbox(polygon)
    image = image.crop(bbox)
    x1, y1, x2, y2 = bbox

    # Adjust polygon so that its coordinates are in respect to the cropped image.
    polygon = [(x - x1, y - y1) for x, y in polygon]
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    mask = cv2.fillPoly(mask, np.array([polygon], dtype=np.int32), color=255)
    mask = Image.fromarray(mask, mode="L")
    fill = Image.new(image.mode, image.size, color=(255, 255, 255))
    return Image.composite(image, fill, mask)


def pages_from_path(path):
    pbar = tqdm(desc=f"Scanning '{path}'", unit=" PageXMLs")
    pages = []
    for parent, dirs, files in os.walk(path):
        if "page" not in dirs:
            continue

        for file in files:
            xml_path = os.path.join(parent, "page", Path(file).with_suffix(".xml"))
            if not os.path.exists(xml_path):
                continue
            image_path = os.path.join(parent, file)
            pages.append(Page(xml_path, image_path))
            pbar.update(1)

    return pages
