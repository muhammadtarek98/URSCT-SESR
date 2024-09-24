from model.URSCT_model import URSCT
import torch
ckpt="/home/cplus/projects/m.tarek_master/Image_enhancement/weights/URSCT-SESR/model_bestSSIM.pth"
state_dict=torch.load(f=ckpt)
opt={"IN_CHANS":3,
"OUT_CHANS":3,
"IMG_SIZE":[256, 256],
"PATCH_SIZE":2,
"WIN_SIZE":8,
"EMB_DIM":32,
"DEPTH_EN":[8, 8, 8, 8],
"HEAD_NUM":[8, 8, 8, 8],
"MLP_RATIO":4.0,
"QKV_BIAS":True,
"QK_SCALE": 8,
"DROP_RATE":0,
"ATTN_DROP_RATE":0.,
"DROP_PATH_RATE":0.1,
"APE": False,
"PATCH_NORM":True,
"USE_CHECKPOINTS":True}
model=URSCT(opt
)
model.load_state_dict(state_dict=state_dict["state_dict"])