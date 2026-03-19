"""
Scripts for preparing training samples from Page XML GT
"""

from PIL.Image import Image as ImageT
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import unicodedata
import os
from pathlib import Path
from tqdm import tqdm
import re


from params import DATA_STOPWORDS


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
            if content is None:
                continue

            # Only accept empty lines from the 'tomma sidor' subset!
            # Other GT may have empty lines but those are likely to be annotation errors.
            if content.text is None and not "tomma_sidor" in self.xml_path:
                continue

            text = content.text or ""
            text = normalize(text)

            try:
                masked_image = apply_mask(image, polygon)
            except ValueError:
                continue

            if reject(image, text):
                continue

            line_id = f"{self.key}_line{i}"
            results.append((line_id, masked_image, text))
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


def normalize(text: str) -> str:
    """
    Normalize `text`
    """
    fractions = [
        ("½", "1/2"),
        ("⅐", "1/7"),
        ("⅑", "1/9"),
        ("⅒", "1/10"),
        ("⅓", "1/3"),
        ("⅔", "2/3"),
        ("⅕", "1/5"),
        ("⅖", "2/5"),
        ("⅗", "3/5"),
        ("⅘", "4/5"),
        ("⅙", "1/6"),
        ("⅚", "5/6"),
        ("⅛", "1/8"),
        ("⅜", "3/8"),
        ("⅝", "5/8"),
        ("⅞", "7/8"),
    ]

    for a, b in fractions:
        # in mixed whole- and fraction numbers, there might not be a space between
        # the whole part and the fraction part, like this: 2½
        # when replacing the fractions, we need to insert a whitespace to preserve
        # the original number:  2½ -> 2 1/2 (and not 21/2!)
        # if there was a space (2 ½) this causes double whitespaces, but those will be
        # collapsed later on.
        text = text.replace(a, f" {b}")

    # No space before '¬', no repeated '¬'s
    text = re.sub(r"\s*¬", "¬", text)
    text = re.sub(r"¬*", r"¬", text)

    # Remove „
    text = text.replace("„", "")

    # Normalize fancy quotes
    text = text.replace("”", '"').replace("“", '"').replace("´", "'")

    # Make '·' and '‧' into regular '.'
    text = text.replace("·", ".").replace("‧", ".")

    # No tildes
    text = text.replace("~", "")

    # Always put a whitespace after '.'
    # 'Den .... 13 ..Januarii' => 'Den .... 13 .. Januarii'
    text = re.sub(r"\.(\w)", r". \1", text)

    # Replace more than three repeated punctuation marks or '…' with '...'
    # 'Den .... 13 .. Januarii' => 'Den ... 13 .. Januarii'
    text = re.sub(r"(\. ?)(\. ?)(\. ?)+", "...", text)
    text = text.replace("…", "...")

    # Replace '﹘' ('small em dash') with its 'large' version
    text = text.replace("﹘", " — ")

    # Replace all types of dashes with a em dash, IF they're surrounded by whitespace
    text = re.sub(r" (-|_|—|‒|―|–) ", " — ", text)

    # Replace repeated dashes, dots or similar with a single em dash (TOGMF-style)
    # 'Carl _ _ _ Wikman. 16.' => 'Carl — Wikman. 16.'
    text = re.sub(r"([^\w\.\",])(\W|_)+([^\w\.\"])", " — ", text)

    # Replace all (possibly repeated) whitespace-like characters with a singe ' '
    text = re.sub(r"\s+", " ", text)

    # No leading or training whitespace
    text = text.strip()
    text = unicodedata.normalize("NFKC", text)
    return text


def reject(image, text):
    """
    Return True if sample should be rejected.
    """
    contains_stopwords = any(stopword in text for stopword in DATA_STOPWORDS)
    too_small = image.height < 10 or image.width < 10
    return contains_stopwords or too_small
