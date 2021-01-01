import pybullet as p
import numpy as np
from gym import spaces
from diy_gym.model import Model
from diy_gym.addons.addon import Addon
import torchvision.models
from PIL import Image
import torch
import logging
from torchvision.transforms.functional import to_tensor


class Camera(Addon):
    """Captures RGB, depth and segmentation images from the perspective of a model or fixed in the world frame.

    Configs:
        clipping_boundaries (list of float, optional, [0.1, 100]): the boundaries in the z-direction of the volume
            captured by the depth images (i.e objects at a distance that is outside these boundaries won't
            appear in the depth image)
        field_of_view (float, optional, 70): defines the perspective matrix of the camera
        frame (str, optional): the frame on the parent model to which the camera will be attached
            (defaults to the base frame)
        resolution (list of float, optional, [640, 480]): defines the width and height (respectively) of the
            images captured by the camera
        xyz (list of float, optional, [0,0,0]): the offset relative to the `frame` at which to spawn the camera
        rpy (list of float, optional, [0,0,0]): the orientation relative to the `frame` at which to spawn the camera
            expressed in euler angles
        use_depth (bool, optional, False): whether to include depth images in observations
        use_segmentation_mask (bool, optional, False): whether to include segmentation images in observations
        use_features (bool, optional, False): Use resnet to extract features (must be 256, 256)
    """
    def __init__(self, parent, config):
        super(Camera, self).__init__(parent, config)

        self.near, self.far = config.get('clipping_boundaries', [0.01, 100])
        self.fov = config.get('field_of_view', 70.0)
        self.resolution = config.get('resolution', [256, 256])
        self.aspect = self.resolution[0] / self.resolution[1]

        self.uid = parent.uid if isinstance(parent, Model) else -1
        self.frame_id = parent.get_frame_id(config.get('frame')) if 'frame' in config else -1

        xyz = config.get('xyz', [0., 0., 0.])
        rpy = config.get('rpy', [0., 0., 0.])

        self.use_depth = config.get('use_depth', True)
        self.use_seg_mask = config.get('use_segmentation_mask', False)
        self.use_features = config.get('use_features', False)

        self.T_parent_cam = self.trans_from_xyz_quat(xyz, p.getQuaternionFromEuler(rpy))
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)
        self.K = np.array(self.projection_matrix).reshape([4, 4]).T

        self.observation_space = spaces.Dict({
            'rgb': spaces.Box(0., 1., shape=self.resolution + [3], dtype='float32'),
        })

        if self.use_depth:
            self.observation_space.spaces.update({'depth': spaces.Box(0., 10., shape=self.resolution, dtype='float32')})

        if self.use_features:
            resnet18 = torchvision.models.resnet18(pretrained=True).eval()
            modules=list(resnet18.children())[:-1]
            self.feature_extractor = torch.nn.Sequential(*modules).eval()
            self.observation_space.spaces.update(
                {'features': spaces.Box(-100., 100., shape=(512* 2,) if self.use_depth else(512,), dtype='float32')})
            self.use_cuda = torch.cuda.is_available()
            if self.use_cuda:
                logging.info('using cuda.half for feature extraction')
                self.feature_extractor.to('cuda').half()
        
        if self.use_seg_mask:
            self.observation_space.spaces.update(
                {'segmentation_mask': spaces.Box(0., 10., shape=self.resolution, dtype='float32')})

    def extract_features(self, obs):
        # see https://arxiv.org/pdf/1710.10710.pdf

        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        with torch.no_grad():
            im = Image.fromarray(np.uint8(obs['rgb'] * 255))
            im = im.resize((224, 224))
            x1 = normalize(to_tensor(im))

            im = Image.fromarray(np.uint8(obs['depth'] * 255)).convert('RGB')
            im = im.resize((224, 224))
            x2 = normalize(to_tensor(im))

            x = torch.stack([x1, x2])
            if self.use_cuda:
                x = x.to('cuda').half()
            h = self.feature_extractor(x)
            h = h.cpu().detach().numpy() # shape (2, 512)
        return h.flatten()
        

    def observe(self):
        if self.frame_id >= 0:
            link_state = p.getLinkState(self.uid, self.frame_id)
            T_world_parent = self.trans_from_xyz_quat(link_state[4], link_state[5])
        elif self.uid >= 0:
            model_state = p.getBasePositionAndOrientation(self.uid)
            T_world_parent = self.trans_from_xyz_quat(model_state[0], model_state[1])
        else:
            T_world_parent = np.eye(4)

        T_world_cam = np.linalg.inv(T_world_parent.dot(self.T_parent_cam))

        image = p.getCameraImage(self.resolution[0],
                                 self.resolution[1],
                                 T_world_cam.T.flatten(),
                                 self.projection_matrix,
                                 flags=p.ER_NO_SEGMENTATION_MASK if not self.use_seg_mask else 0,
                                 shadow=False,
                                #  renderer=p.ER_TINY_RENDERER,
                                #  renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                 )

        obs = {
            'rgb': np.array(image[2], copy=False).reshape(self.resolution + [4])[:, :, :3] / 255.
        }  # discard the alpha channel and normalise to [0 1]

        if self.use_depth:
            # the depth buffer is normalised to [0 1] whereas NDC coords require [-1 1] ref: https://bit.ly/2rcXidZ
            depth_ndc = np.array(image[3], copy=False).reshape(self.resolution) * 2 - 1

            # recover eye coordinate depth using the projection matrix ref: https://bit.ly/2vZJCsx
            depth = self.K[2, 3] / (self.K[3, 2] * depth_ndc - self.K[2, 2])

            obs['depth'] = depth

        if self.use_seg_mask:
            obs['segmentation_mask'] = np.array(image[4], copy=False).reshape(self.resolution)

        if self.use_features:
            obs['features'] = self.extract_features(obs)

        return obs

    def trans_from_xyz_quat(self, xyz, quat):
        T = np.eye(4)
        T[:3, 3] = xyz
        T[:3, :3] = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        return T
