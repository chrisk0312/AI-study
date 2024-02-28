# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
PyTorch Hub models https://pytorch.org/hub/ultralytics_yolov5

Usage:
    import torch #torch 사용
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # official model
    model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s')  # from branch
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s.pt')  # custom/local model
    model = torch.hub.load('.', 'custom', 'yolov5s.pt', source='local')  # local repo
"""

import torch #torch 사용


def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None): #모델 생성
    """
    Creates or loads a YOLOv5 model. #YOLOv5 모델을 생성하거나 로드합니다.

    Arguments:#인수
        name (str): model name 'yolov5s' or path 'path/to/best.pt' #모델 이름
        pretrained (bool): load pretrained weights into the model #사전 학습된 가중치를 모델에 로드
        channels (int): number of input channels #입력 채널 수
        classes (int): number of model classes #모델 클래스 수
        autoshape (bool): apply YOLOv5 .autoshape() wrapper to model #YOLOv5 .autoshape() 래퍼를 모델에 적용
        verbose (bool): print all information to screen #모든 정보를 화면에 출력
        device (str, torch.device, None): device to use for model parameters #모델 매개변수에 사용할 장치

    Returns:#반환
        YOLOv5 model
    """
    from pathlib import Path #경로 설정

    from models.common import AutoShape, DetectMultiBackend #모델 공통
    from models.experimental import attempt_load #실험적 모델
    from models.yolo import ClassificationModel, DetectionModel, SegmentationModel #yolo 모델
    from utils.downloads import attempt_download #다운로드 시도
    from utils.general import LOGGER, ROOT, check_requirements, intersect_dicts, logging #일반적인 유틸리티
    from utils.torch_utils import select_device #torch 유틸리티

    if not verbose: #verbose가 아니면
        LOGGER.setLevel(logging.WARNING) #로거 레벨을 경고로 설정
    check_requirements(ROOT / "requirements.txt", exclude=("opencv-python", "tensorboard", "thop")) #요구 사항 확인
    name = Path(name) #경로 설정
    path = name.with_suffix(".pt") if name.suffix == "" and not name.is_dir() else name #경로 설정
    try: #시도
        device = select_device(device) #장치 선택
        if pretrained and channels == 3 and classes == 80: #사전 학습 및 채널이 3이고 클래스가 80이면
            try: #시도
                model = DetectMultiBackend(path, device=device, fuse=autoshape)  # load DetectMultiBackend
                if autoshape: #자동 모양이면
                    if model.pt and isinstance(model.model, ClassificationModel): #모델이 분류 모델이고
                        LOGGER.warning( #경고
                            "WARNING ⚠️ YOLOv5 ClassificationModel is not yet AutoShape compatible. " #경고
                            "You must pass torch tensors in BCHW to this model, i.e. shape(1,3,224,224)." #이 모델에는 torch 텐서를 전달해야 합니다. 즉, 모양(1,3,224,224).
                        )
                    elif model.pt and isinstance(model.model, SegmentationModel):# 모델이 세분화 모델이고
                        LOGGER.warning( #경고
                            "WARNING ⚠️ YOLOv5 SegmentationModel is not yet AutoShape compatible. " #경고
                            "You will not be able to run inference with this model." #이 모델로 추론을 실행할 수 없습니다.
                        )
                    else: #그렇지 않으면
                        model = AutoShape(model) # add .autoshape() wrapper #.autoshape() 래퍼 추가, 모델을 자동 모양으로 설정
            except Exception: # not a .pt file #.pt 파일이 아닌 경우
                model = attempt_load(path, device=device, fuse=False)  # load FP32 model #FP32 모델 로드
        else: #그렇지 않으면
            cfg = list((Path(__file__).parent / "models").rglob(f"{path.stem}.yaml"))[0]  # model.yaml 경로
            model = DetectionModel(cfg, channels, classes)  # create #생성
            if pretrained: #사전 학습이면
                ckpt = torch.load(attempt_download(path), map_location=device)  # load FP32 model #FP32 모델 로드
                csd = ckpt["model"].float().state_dict()  # to FP32 #FP32로 변환
                csd = intersect_dicts(csd, model.state_dict(), exclude=["anchors"])  # intersect #교차
                model.load_state_dict(csd, strict=False)  # load #로드
                if len(ckpt["model"].names) == classes:#클래스 수가 같으면
                    model.names = ckpt["model"].names  # set class names #클래스 이름 설정
        if not verbose: #verbose가 아니면
            LOGGER.setLevel(logging.INFO)  # reset to INFO #INFO로 재설정
        return model.to(device) #장치로 반환

    except Exception as e: #예외
        help_url = "https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading" #도움말 URL
        s = f"{e}. Cache may be out of date, try `force_reload=True` or see {help_url} for help." #캐시가 오래되었을 수 있습니다. `force_reload=True`를 시도하거나 도움말을 참조하십시오.
        raise Exception(s) from e #예외 발생


def custom(path="path/to/model.pt", autoshape=True, _verbose=True, device=None): #사용자 정의 모델
    # YOLOv5 custom or local model
    return _create(path, autoshape=autoshape, verbose=_verbose, device=device) #생성


def yolov5n(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None): #모델 생성
    # YOLOv5-nano model https://github.com/ultralytics/yolov5
    return _create("yolov5n", pretrained, channels, classes, autoshape, _verbose, device) #생성


def yolov5s(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None): #모델 생성
    # YOLOv5-small model https://github.com/ultralytics/yolov5
    return _create("yolov5s", pretrained, channels, classes, autoshape, _verbose, device) #생성


def yolov5m(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None): #모델 생성
    # YOLOv5-medium model https://github.com/ultralytics/yolov5
    return _create("yolov5m", pretrained, channels, classes, autoshape, _verbose, device) #생성


def yolov5l(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None): #모델 생성
    # YOLOv5-large model https://github.com/ultralytics/yolov5
    return _create("yolov5l", pretrained, channels, classes, autoshape, _verbose, device) #생성


def yolov5x(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None): #모델 생성
    # YOLOv5-xlarge model https://github.com/ultralytics/yolov5
    return _create("yolov5x", pretrained, channels, classes, autoshape, _verbose, device) #생성


def yolov5n6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None): #모델 생성
    # YOLOv5-nano-P6 model https://github.com/ultralytics/yolov5
    return _create("yolov5n6", pretrained, channels, classes, autoshape, _verbose, device) #생성


def yolov5s6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None): #모델 생성
    # YOLOv5-small-P6 model https://github.com/ultralytics/yolov5
    return _create("yolov5s6", pretrained, channels, classes, autoshape, _verbose, device)#생성


def yolov5m6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None): #모델 생성
    # YOLOv5-medium-P6 model https://github.com/ultralytics/yolov5
    return _create("yolov5m6", pretrained, channels, classes, autoshape, _verbose, device) #생성


def yolov5l6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None): #모델 생성
    # YOLOv5-large-P6 model https://github.com/ultralytics/yolov5
    return _create("yolov5l6", pretrained, channels, classes, autoshape, _verbose, device) #생성


def yolov5x6(pretrained=True, channels=3, classes=80, autoshape=True, _verbose=True, device=None): #모델 생성
    # YOLOv5-xlarge-P6 model https://github.com/ultralytics/yolov5
    return _create("yolov5x6", pretrained, channels, classes, autoshape, _verbose, device) #생성


if __name__ == "__main__": #메인 함수
    import argparse #argparse 사용
    from pathlib import Path #경로 설정

    import numpy as np #numpy 사용
    from PIL import Image #PIL 사용

    from utils.general import cv2, print_args #일반적인 유틸리티

    # Argparser
    parser = argparse.ArgumentParser() #인수 구문 분석기
    parser.add_argument("--model", type=str, default="yolov5s", help="model name") #모델 이름
    opt = parser.parse_args() #인수 구문 분석기 구문 분석
    print_args(vars(opt))#인수 출력

    # Model
    model = _create(name=opt.model, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True) #모델 생성
    # model = custom(path='path/to/model.pt')  # custom

    # Images
    imgs = [ #이미지
        "data/images/zidane.jpg",  # filename
        Path("data/images/zidane.jpg"),  # Path
        "https://ultralytics.com/images/zidane.jpg",  # URI
        cv2.imread("data/images/bus.jpg")[:, :, ::-1],  # OpenCV
        Image.open("data/images/bus.jpg"), # PIL(Python Imaging Library)
        np.zeros((320, 640, 3)), #
    ]  # numpy

    # Inference
    results = model(imgs, size=320)  # batched inference

    # Results
    results.print()
    results.save()
