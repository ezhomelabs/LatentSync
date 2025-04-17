# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torchvision import transforms
import cv2
from einops import rearrange
import mediapipe as mp
import torch
import numpy as np
from typing import Union
from .affine_transform import AlignRestore, laplacianSmooth
import face_alignment

"""
If you are enlarging the image, you should prefer to use INTER_LINEAR or INTER_CUBIC interpolation. If you are shrinking the image, you should prefer to use INTER_AREA interpolation.
https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
"""


def load_fixed_mask(resolution: int, mask_image_path="latentsync/utils/mask.png") -> torch.Tensor:
    mask_image = cv2.imread(mask_image_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.resize(mask_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4) / 255.0
    mask_image = rearrange(torch.from_numpy(mask_image), "h w c -> c h w")
    return mask_image


class ImageProcessor:
    def __init__(self, resolution: int = 512, mask: str = "fix_mask", device: str = "cpu", mask_image=None):
        self.resolution = resolution
        self.resize = transforms.Resize(
            (resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
        )
        self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)
        self.mask = mask

        if mask in ["mouth", "face", "eye"]:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)  # Process single image
        if mask == "fix_mask":
            self.face_mesh = None
            self.smoother = laplacianSmooth()
            self.restorer = AlignRestore()

            if mask_image is None:
                self.mask_image = load_fixed_mask(resolution)
            else:
                self.mask_image = mask_image

            if device != "cpu":
                self.fa = face_alignment.FaceAlignment(
                    face_alignment.LandmarksType.TWO_D, flip_input=False, device=device
                )
                self.face_mesh = None
            else:
                # self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)  # Process single image
                self.face_mesh = None
                self.fa = None

    def detect_facial_landmarks(self, image: np.ndarray):
        height, width, _ = image.shape
        results = self.face_mesh.process(image)
        if not results.multi_face_landmarks:  # Face not detected
            raise RuntimeError("Face not detected")
        face_landmarks = results.multi_face_landmarks[0]  # Only use the first face in the image
        landmark_coordinates = [
            (int(landmark.x * width), int(landmark.y * height)) for landmark in face_landmarks.landmark
        ]  # x means width, y means height
        return landmark_coordinates

    def preprocess_one_masked_image(self, image: torch.Tensor) -> np.ndarray:
        image = self.resize(image)

        if self.mask == "mouth" or self.mask == "face":
            landmark_coordinates = self.detect_facial_landmarks(image)
            if self.mask == "mouth":
                surround_landmarks = mouth_surround_landmarks
            else:
                surround_landmarks = face_surround_landmarks

            points = [landmark_coordinates[landmark] for landmark in surround_landmarks]
            points = np.array(points)
            mask = np.ones((self.resolution, self.resolution))
            mask = cv2.fillPoly(mask, pts=[points], color=(0, 0, 0))
            mask = torch.from_numpy(mask)
            mask = mask.unsqueeze(0)
        elif self.mask == "half":
            mask = torch.ones((self.resolution, self.resolution))
            height = mask.shape[0]
            mask[height // 2 :, :] = 0
            mask = mask.unsqueeze(0)
        elif self.mask == "eye":
            mask = torch.ones((self.resolution, self.resolution))
            landmark_coordinates = self.detect_facial_landmarks(image)
            y = landmark_coordinates[195][1]
            mask[y:, :] = 0
            mask = mask.unsqueeze(0)
        else:
            raise ValueError("Invalid mask type")

        image = image.to(dtype=torch.float32)
        pixel_values = self.normalize(image / 255.0)
        masked_pixel_values = pixel_values * mask
        mask = 1 - mask

        return pixel_values, masked_pixel_values, mask

    def affine_transform(self, image: torch.Tensor, allow_multi_faces: bool = True) -> np.ndarray:
        """
        Align a face or, if none detected, fall back to a central crop.
        Returns:
          face_tensor (torch.Tensor): C×H×W aligned face (or crop) 
          box (list[int]):  [x1, y1, x2, y2] in the original image
          affine_matrix (np.ndarray): 3×3 transform (identity on fallback)
        """
        # bring image to H×W×C numpy if needed
        img_np = image.numpy().transpose(1, 2, 0) if isinstance(image, torch.Tensor) else image.copy()

        # --- try real face detection + alignment ---
        try:
            if self.fa is None:
                # Mediapipe path
                landmark_coordinates = np.array(self.detect_facial_landmarks(img_np))
                lm68 = mediapipe_lm478_to_face_alignment_lm68(landmark_coordinates)
            else:
                # face_alignment path
                detected = self.fa.get_landmarks(img_np)
                if detected is None:
                    raise RuntimeError("Face not detected")
                if not allow_multi_faces and len(detected) > 1:
                    raise RuntimeError("More than one face detected")
                lm68 = detected[0]

            # smooth & pick three anchor points
            pts = self.smoother.smooth(lm68)
            lmk3_ = np.vstack([
                pts[17:22].mean(0),
                pts[22:27].mean(0),
                pts[27:36].mean(0),
            ])

            # perform the actual align & warp
            face, affine_matrix = self.restorer.align_warp_face(
                img_np.copy(), lmks3=lmk3_, smooth=True, border_mode="constant"
            )
            box = [0, 0, face.shape[1], face.shape[0]]
            face_resized = cv2.resize(face, (self.resolution, self.resolution),
                                      interpolation=cv2.INTER_LANCZOS4)
            face_tensor = rearrange(torch.from_numpy(face_resized), "h w c -> c h w")
            return face_tensor, box, affine_matrix

        except RuntimeError:
            # --- fallback: no face detected, use central crop + identity matrix ---
            h, w, _ = img_np.shape
            # define a central box (you can tweak the fraction here)
            x1, y1 = w // 4, h // 4
            x2, y2 = x1 * 3, y1 * 3
            crop = img_np[y1:y2, x1:x2]
            resized = cv2.resize(crop, (self.resolution, self.resolution),
                                 interpolation=cv2.INTER_LANCZOS4)
            face_tensor = rearrange(torch.from_numpy(resized), "h w c -> c h w")
            box = [x1, y1, x2, y2]
            affine_matrix = np.eye(3, dtype=np.float32)
            return face_tensor, box, affine_matrix

    def preprocess_fixed_mask_image(self, image: torch.Tensor, affine_transform=False):
        if affine_transform:
            image, _, _ = self.affine_transform(image)
        else:
            image = self.resize(image)
        pixel_values = self.normalize(image / 255.0)
        masked_pixel_values = pixel_values * self.mask_image
        return pixel_values, masked_pixel_values, self.mask_image[0:1]

    def prepare_masks_and_masked_images(self, images: Union[torch.Tensor, np.ndarray], affine_transform=False):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")
        if self.mask == "fix_mask":
            results = [self.preprocess_fixed_mask_image(image, affine_transform=affine_transform) for image in images]
        else:
            results = [self.preprocess_one_masked_image(image) for image in images]

        pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
        return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), torch.stack(masks_list)

    def process_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")
        images = self.resize(images)
        pixel_values = self.normalize(images / 255.0)
        return pixel_values

    def close(self):
        if self.face_mesh is not None:
            self.face_mesh.close()


def mediapipe_lm478_to_face_alignment_lm68(lm478, return_2d=True):
    """
    lm478: [B, 478, 3] or [478,3]
    """
    # lm478[..., 0] *= W
    # lm478[..., 1] *= H
    landmarks_extracted = []
    for index in landmark_points_68:
        x = lm478[index][0]
        y = lm478[index][1]
        landmarks_extracted.append((x, y))
    return np.array(landmarks_extracted)


landmark_points_68 = [
    162,
    234,
    93,
    58,
    172,
    136,
    149,
    148,
    152,
    377,
    378,
    365,
    397,
    288,
    323,
    454,
    389,
    71,
    63,
    105,
    66,
    107,
    336,
    296,
    334,
    293,
    301,
    168,
    197,
    5,
    4,
    75,
    97,
    2,
    326,
    305,
    33,
    160,
    158,
    133,
    153,
    144,
    362,
    385,
    387,
    263,
    373,
    380,
    61,
    39,
    37,
    0,
    267,
    269,
    291,
    405,
    314,
    17,
    84,
    181,
    78,
    82,
    13,
    312,
    308,
    317,
    14,
    87,
]


# Refer to https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
mouth_surround_landmarks = [
    164,
    165,
    167,
    92,
    186,
    57,
    43,
    106,
    182,
    83,
    18,
    313,
    406,
    335,
    273,
    287,
    410,
    322,
    391,
    393,
]

face_surround_landmarks = [
    152,
    377,
    400,
    378,
    379,
    365,
    397,
    288,
    435,
    433,
    411,
    425,
    423,
    327,
    326,
    94,
    97,
    98,
    203,
    205,
    187,
    213,
    215,
    58,
    172,
    136,
    150,
    149,
    176,
    148,
]

if __name__ == "__main__":
    image_processor = ImageProcessor(512, mask="fix_mask")
    video = cv2.VideoCapture("assets/demo1_video.mp4")
    while True:
        ret, frame = video.read()
        # if not ret:
        #     break

        # cv2.imwrite("image.jpg", frame)

        frame = rearrange(torch.Tensor(frame).type(torch.uint8), "h w c ->  c h w")
        # face, masked_face, _ = image_processor.preprocess_fixed_mask_image(frame, affine_transform=True)
        face, _, _ = image_processor.affine_transform(frame)

        break

    face = (rearrange(face, "c h w -> h w c").detach().cpu().numpy()).astype(np.uint8)
    cv2.imwrite("face.jpg", face)

    # masked_face = (rearrange(masked_face, "c h w -> h w c").detach().cpu().numpy()).astype(np.uint8)
    # cv2.imwrite("masked_face.jpg", masked_face)
