import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def image_to_text(way):
    img = Image.open(f'{way}')
    text = pytesseract.image_to_string(img, lang="rus")
    return text