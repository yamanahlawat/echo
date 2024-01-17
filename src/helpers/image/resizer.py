from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace
from PIL import ExifTags, Image


class FaceDetectorModel(Enum):
    RETINAFACE = "retinaface"
    OPENCV = "opencv"
    MTCNN = "mtcnn"
    SSD = "ssd"
    DLIB = "dlib"


class ImagePreprocessing:
    ASPECT_RATIOS = {
        (512, 512),
        (512, 768),
        (768, 512),
        (1024, 768),
        (768, 1024),
        (1024, 1024),
    }

    def load_image(self, image_path: Path) -> Image.Image:
        """
        Load an image from the given `image_path` and correct its orientation based on EXIF data.
        Args:
            image_path (Path): The path to the image.
        Raises:
            ValueError: If the `image_path` does not exist.
        Returns:
            Image.Image: The loaded and oriented image.
        """
        if not image_path.exists():
            raise ValueError("Image doesn't exists")
        image = Image.open(image_path)
        image = self.orientation(image=image)
        return image

    def detect_faces(
        self,
        image_input: Path | np.ndarray | str | Image.Image,
        detector_model: str = FaceDetectorModel.RETINAFACE.value,
    ) -> list[dict]:
        """
        Detect faces in the given image using the DeepFace model.
        Args:
            image_input (Path | np.ndarray | str | Image.Image): The input image.
                Can be a file path, a NumPy array, a base64 encoded string, or a PIL Image object.
            detector_model (str, optional): The backend model to use for face detection.
                Defaults to FaceDetectorModel.RETINAFACE.value.
        Returns:
            list[dict]: A list of dictionaries, each containing information about a detected face.
        Raises:
            ValueError: If no faces are detected in the image.
        """
        if isinstance(image_input, Image.Image):
            numpy_image = np.array(image_input)
            bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            image_input = bgr_image
        # try with retinaface, if not found try with opencv
        try:
            results = DeepFace.extract_faces(img_path=image_input, detector_backend=detector_model)
        except ValueError:
            results = DeepFace.extract_faces(img_path=image_input, detector_backend=FaceDetectorModel.OPENCV.value)
        if not results:
            raise ValueError("No faces detected")
        return results

    def orientation(self, image):
        """
        Correct the orientation of the image based on its EXIF data.
        Args:
            image (Image.Image): The image to correct.
        Returns:
            Image.Image: The corrected image.
        """
        try:
            exif = {
                ExifTags.TAGS[k]: v
                for k, v in image.getexif().items()
                if k in ExifTags.TAGS and isinstance(v, (int, bytes))
            }
        except AttributeError:
            exif = {}
        orientation = exif.get("Orientation", 1)

        if orientation == 1:
            # Normal, do nothing
            pass
        elif orientation == 2:
            # Flipped horizontally
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Upside down
            image = image.rotate(180, expand=True)
        elif orientation == 4:
            # Flipped vertically
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            # Rotated 90 deg CCW and flipped
            image = image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 deg CCW
            image = image.rotate(-90, expand=True)
        elif orientation == 7:
            # Rotated 90 deg CW and flipped
            image = image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 90 deg CW
            image = image.rotate(90, expand=True)

        return image

    def expand_box(
        self,
        image: Image.Image,
        box: tuple[int, int, int, int],
        target_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        """
        Expand the given box in the image to the largest possible size with the target aspect ratio.
        Args:
            image (Image.Image): The image containing the box.
            box (tuple[int, int, int, int]): The box coordinates as (x1, y1, width, height).
            target_size (tuple[int, int]): The target size as (width, height).
        Returns:
            tuple[int, int, int, int]: The expanded box coordinates as (x1, y1, x2, y2).
        """
        # Unpack the face box coordinates
        x1, y1, width, height = box

        # Calculate target aspect ratio
        target_aspect_ratio = target_size[0] / target_size[1]

        # Calculate the maximum size rectangle we can fit in the image with the given aspect ratio
        if image.width / image.height > target_aspect_ratio:
            # The image is wider than the target aspect ratio,
            # the height of the rectangle will be the height of the image
            rectangle_height = image.height
            rectangle_width = rectangle_height * target_aspect_ratio
        else:
            # The image is taller than the target aspect ratio,
            # the width of the rectangle will be the width of the image
            rectangle_width = image.width
            rectangle_height = rectangle_width / target_aspect_ratio

        # Calculate the center of the box
        center_x = x1 + width / 2
        center_y = y1 + height / 2

        # Calculate the top left and bottom right coordinates of the rectangle
        # After calculating new_x1 and new_y1 based on the center of the box
        new_x1 = max(0, int(center_x - rectangle_width / 2))
        new_y1 = max(0, int(center_y - rectangle_height / 2))

        # Calculate the width and height based on new_x1 and new_y1
        actual_width = min(rectangle_width, image.width - new_x1)

        # Calculate new_x2 and new_y2 based on actual_width and actual_height while maintaining aspect ratio
        new_x2 = new_x1 + actual_width
        new_y2 = new_y1 + int(actual_width / target_aspect_ratio)  # Keeping aspect ratio intact

        # Ensure new_y2 doesn't exceed image boundaries
        new_y2 = min(new_y2, image.height)

        return new_x1, new_y1, new_x2, new_y2

    def resize_image(self, image_path: Path, target_size: tuple[int, int]) -> Image.Image:
        """
        Resize an image to the target size, maintaining aspect ratio and detecting faces.
        Args:
            image_path (Path): Path to the image file.
            target_size (tuple[int, int]): The target size for the image.
        Returns:
            Image.Image: The resized image.
        Raises:
            ValueError: If the target size is not in the allowed aspect ratios.
        """
        if target_size not in self.ASPECT_RATIOS:
            raise ValueError(f"Target size {target_size} is not in the allowed aspect ratios")
        image = self.load_image(image_path=image_path)
        results = self.detect_faces(image_input=image)
        # get the first face
        first_face = results[0]
        facial_area = first_face["facial_area"]
        face_box = (
            facial_area["x"],
            facial_area["y"],
            facial_area["w"],
            facial_area["h"],
        )
        face_box = self.expand_box(
            image=image,
            box=face_box,
            target_size=target_size,
        )
        image = image.crop(face_box)
        return image.resize(target_size)
