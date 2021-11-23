import os
import numpy as np
import cv2
import torch
from pytorch3d.structures import Meshes
from pytorch3d.io import IO
from pytorch3d.renderer import (
    look_at_view_transform,
    look_at_rotation,
    OpenGLPerspectiveCameras,
    PerspectiveCameras,
    PointLights,
    DirectionalLights,
    AmbientLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesVertex,
    BlendParams,
)

# Params
K = np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]])

f_x, f_y = K[0, 0], K[1, 1]
p_x, p_y = K[0, 2], K[1, 2]
h = 480
w = 640

# Load mesh
device = torch.device("cuda:0")
mesh = IO().load_mesh("lamp.ply").to(device)
mesh.scale_verts_(0.001)

# import ipdb; ipdb.set_trace()

# GT Pose for instance 176
R = torch.tensor(
    [
        [0.66307002, 0.74850100, 0.00921593],
        [0.50728703, -0.44026601, -0.74082798],
        [-0.55045301, 0.49589601, -0.67163098],
    ],
    dtype=torch.float32,
)
T = torch.tensor([42.36749640, 1.84263252, 768.28001229], dtype=torch.float32) / 1000

# Apply fix #294
RT = torch.zeros((4, 4))
RT[3, 3] = 1
RT[:3, :3] = R
RT[:3, 3] = T

Rz = torch.tensor([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).float()

RT = torch.matmul(Rz, RT)

R = RT[:3, :3].t().reshape(1, 3, 3)
T = RT[:3, 3].reshape(1, 3)

f = torch.tensor((f_x, f_y), dtype=torch.float32).unsqueeze(0)
p = torch.tensor((p_x, p_y), dtype=torch.float32).unsqueeze(0)
img_size = torch.tensor((h, w), dtype=torch.float32).unsqueeze(0)

lights = AmbientLights(device=device)

camera = PerspectiveCameras(
    R=R, T=T, focal_length=f, principal_point=p, image_size=((h, w),), device=device, in_ndc=False
)

blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
# Set Renderer Parameters
raster_settings = RasterizationSettings(
    image_size=(h, w),
    blur_radius=0.0,
    faces_per_pixel=1,
    max_faces_per_bin=mesh.faces_packed().shape[0],
    perspective_correct=True,
)

rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings)

renderer = MeshRenderer(
    rasterizer,
    shader=SoftPhongShader(
        device=device,
        cameras=camera,
        lights=lights,
        blend_params=blend_params,
    ),
)

# Generate rendered image
target_images = renderer(mesh, cameras=camera, lights=lights)

img = target_images[0, ..., :3]

bg_pth = "lamp_176.png"
bg = cv2.imread(bg_pth, cv2.IMREAD_COLOR)

imgray = cv2.cvtColor(img.cpu().numpy(), cv2.COLOR_BGR2GRAY)
img_255 = (img.cpu().numpy() * 255).astype("uint8")
img_bg = (img_255 * 0.5 + bg * 0.5).astype("uint8")
cv2.imshow("", cv2.hconcat([img_255, bg, img_bg]))
cv2.waitKey(0)

# ret, mask = cv2.threshold(imgray, 1, 255, 0)
mask = (imgray > 0).astype("uint8") * 255
cv2.imshow("mask", mask)
cv2.waitKey(0)

# bg = None
# while bg is None or len(bg.shape) < 3:

#     bg = cv2.cvtColor(cv2.imread(bg_pth), cv2.COLOR_BGR2RGB)
#     if len(bg.shape) < 3:
#         bg = None
#         continue
#     bg_h, bg_w, _ = bg.shape
#     if bg_h < h:
#         new_w = int(float(h) / bg_h * bg_w)
#         bg = cv2.resize(bg, (new_w, h))
#     bg_h, bg_w, _ = bg.shape
#     if bg_w < w:
#         new_h = int(float(w) / bg_w * bg_h)
#         bg = cv2.resize(bg, (w, new_h))
#     bg_h, bg_w, _ = bg.shape
#     if bg_h > h:
#         sh = randint(0, bg_h - h)
#         bg = bg[sh : sh + h, :, :]
#     bg_h, bg_w, _ = bg.shape
#     if bg_w > w:
#         sw = randint(0, bg_w - w)
#         bg = bg[:, sw : sw + w, :]

#     bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)

# msk_3c = np.repeat(mask[:, :, None], 3, axis=2)

# msk_3c = msk_3c>0

# rgb = img.cpu().numpy() * (msk_3c).astype(bg.dtype) + bg * (msk_3c == 0).astype(bg.dtype)

# cv2.imwrite('test.jpg', rgb)
