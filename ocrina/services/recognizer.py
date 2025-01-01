import cv2
import pytesseract
from PIL import Image
from cv2 import Mat, UMat
from numpy import ndarray

from ocrina.shared.utils.logger import Logger

type RawImage = Mat | ndarray | UMat


class Recognizer:
    def __init__(self):
        self.logger = Logger()

    @staticmethod
    def recognize_image(image: Image) -> str:
        text = pytesseract.image_to_string(image)
        return text

    @staticmethod
    def recognize_raw_image(raw_image: RawImage) -> str:
        image = Recognizer.process_raw_image(raw_image)
        text = pytesseract.image_to_string(image)
        return text

    @staticmethod
    def recognize_file(path: str) -> str:
        raw_image = cv2.imread(path)
        image = Recognizer.process_raw_image(raw_image)
        text = pytesseract.image_to_string(image)
        return text

    @staticmethod
    def process_raw_image(raw_image: RawImage) -> Image:
        rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        enhanced_image = Recognizer.enhance_image(rgb_image)
        image = Image.fromarray(enhanced_image)
        return image

    @staticmethod
    def enhance_image(image: RawImage) -> RawImage:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.convertScaleAbs(gray)
        inverted = cv2.bitwise_not(image)
        adaptive_thresh = cv2.adaptiveThreshold(
            inverted,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            21,
            5
        )
        result = cv2.bitwise_not(adaptive_thresh)
        return result
