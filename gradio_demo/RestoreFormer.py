import os

import cv2
import torch
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize

from RestoreFormer_arch import VQVAEGANMultiHeadTransformer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RestoreFormer():
    """Helper for restoration with RestoreFormer.

    It will detect and crop faces, and then resize the faces to 512x512.
    RestoreFormer is used to restored the resized faces.
    The background is upsampled with the bg_upsampler.
    Finally, the faces will be pasted back to the upsample background image.

    Args:
        model_path (str): The path to the GFPGAN model. It can be urls (will first download it automatically).
        upscale (float): The upscale of the final output. Default: 2.
        arch (str): The RestoreFormer architecture. Option: RestoreFormer | RestoreFormer++. Default: RestoreFormer++.
        bg_upsampler (nn.Module): The upsampler for the background. Default: None.
    """

    def __init__(self, model_path, upscale=2, arch='RestoreFromerPlusPlus', bg_upsampler=None, device=None):
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler
        self.arch = arch
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        if arch == 'RestoreFormer':
            self.RF = VQVAEGANMultiHeadTransformer(head_size = 8, ex_multi_scale_num = 0)
        elif arch == 'RestoreFormer++':
            self.RF = VQVAEGANMultiHeadTransformer(head_size = 4, ex_multi_scale_num = 1)
        else:
            raise NotImplementedError(f'Not support arch: {arch}.')
        
        # initialize face helper
        self.face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=self.device,
            model_rootpath=None)
        
        if model_path.startswith('https://'):
            model_path = load_file_from_url(
                url=model_path, model_dir=os.path.join(ROOT_DIR, 'experiments/weights'), progress=True, file_name=None)
        loadnet = torch.load(model_path)
        
        strict=False
        weights = loadnet['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if k.startswith('vqvae.'):
                k = k.replace('vqvae.', '')
            new_weights[k] = v
        self.RF.load_state_dict(new_weights, strict=strict)
        
        self.RF.eval()
        self.RF = self.RF.to(self.device)

    @torch.no_grad()
    def enhance(self, img, has_aligned=False, only_center_face=False, paste_back=True):
        self.face_helper.clean_all()

        if has_aligned:  # the inputs are already aligned
            img = cv2.resize(img, (512, 512))
            self.face_helper.cropped_faces = [img]
        else:
            self.face_helper.read_image(img)
            self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5)
            # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
            # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
            # align and warp each face
            self.face_helper.align_warp_face()

        # face restoration
        for cropped_face in self.face_helper.cropped_faces:
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                output = self.RF(cropped_face_t)[0]
                restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
            except RuntimeError as error:
                print(f'\tFailed inference for RestoreFormer: {error}.')
                restored_face = cropped_face

            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face)

        if not has_aligned and paste_back:
            # upsample the background
            if self.bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]
            else:
                bg_img = None

            self.face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img)
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, restored_img
        else:
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, None
