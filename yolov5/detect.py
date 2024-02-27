# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse #argparse module,  command-line interfaces 작성
import csv #classes가 CSV format으로 tabular data에 읽고 쓰기 작성   to read and write tabular data in CSV forma
import os #os module,  operating system 이 기능적으로 사용하게함 , file system을 읽고 쓰게해줌
import platform # platform module,  platform's hardware, operating system, and interpreter version information를 제공하는 tool
import sys #sys module, 사용되는 변수에 접근가능하게하며, Python interpreter 유지  to functions that interact strongly with the interpreter.
from pathlib import Path #pathlib module에서 Path class를 import함, 파일 시스템 경로를 다루는데 사용

import torch

FILE = Path(__file__).resolve() #Path(__file__)을 resolve(실제 파일 찾기)함
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path: #ROOT가 sys.path에 없으면
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box 
#from ultralytics.utils.plotting에서 Annotator, colors, save_one_box를 import함

from models.common import DetectMultiBackend #models.common에서 DetectMultiBackend를 import함
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams #utils.dataloaders에서 IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams를 import함
from utils.general import ( #utils.general에서 import함
    LOGGER, #used to log messages
    Profile,# 코드를 profile 이용
    check_file, #파일이 존재하는지 확인
    check_img_size, #이미지 사이즈 확인
    check_imshow, # cv2.imshow() 가능하진 확인
    check_requirements, #필요한 패키지가 설치되어있는지 확인
    colorstr, #색상 문자열을 반환
    cv2, #OpenCV
    increment_path, # 경로를 증가시킴
    non_max_suppression,#비최대 억제
    print_args,#인수를 출력
    scale_boxes,#박스 크기 조정
    strip_optimizer,#모델 최적화 제거
    xyxy2xywh,#xyxy2xywh
)
from utils.torch_utils import select_device, smart_inference_mode #utils.torch_utils에서 select_device, smart_inference_mode를 import함


@smart_inference_mode() #smart_inference_mode를 사용하여 모델을 최적화함
def run( #run함수 정의
    weights=ROOT / "yolov5s.pt",  #모델 경로 또는 triton URL
    source=ROOT / "data/images",   #파일/디렉토리/URL/글로브/스크린/0(웹캠)
    data=ROOT / "data/coco128.yaml",   #데이터셋.yaml 경로
    imgsz=(640, 640),   #추론 크기
    conf_thres=0.25,   #신뢰도 임계값 (하나의 변수 x가 어느 값이 되었을 때 특이한 상태나 급격한 변화가 일어나 임계 상태에 있을 때의 x값)
    iou_thres=0.45,   #NMS IOU 임계값
    max_det=1000,  # maximum detections per image #이미지당 최대 감지
    device="",   #cuda 장치, 예를 들어 0 또는 0,1,2,3 또는 cpu
    view_img=False,  #결과 표시
    save_txt=False,   #결과를 *.txt로 저장
    save_csv=False,   #결과를 CSV 형식으로 저장
    save_conf=False,   # --save-txt 레이블에 신뢰도 저장
    save_crop=False,   #잘린 예측 상자 저장
    nosave=False,   #이미지/비디오 저장하지 않음
    classes=None,   #클래스별로 필터링
    agnostic_nms=False,   #클래스에 무관한 NMS
    augment=False,   #증강된 추론
    visualize=False,  ##특징 시각화
    update=False,   #모든 모델 업데이트
    project=ROOT / "runs/detect",   # 결과를 프로젝트/이름에 저장
    name="exp",   #결과를 프로젝트/이름에 저장
    exist_ok=False,   #기존 프로젝트/이름 ok, 증가하지 않음
    line_thickness=3,   #바운딩 박스 두께(픽셀)
    hide_labels=False,  #레이블 숨기기
    hide_conf=False,   #신뢰도 숨기기
    half=False,   #FP16 반정밀도 추론 사용
    dnn=False,   #ONNX 추론에 OpenCV DNN 사용   
    vid_stride=1,   #비디오 프레임 속도 스트라이드
):
    source = str(source) #source를 문자열로 변환
    save_img = not nosave and not source.endswith(".txt")  # inference images 저장
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS) #파일인지 확인
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://")) #URL인지 확인
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file) #웹캠인지 확인
    screenshot = source.lower().startswith("screen") #스크린샷인지 확인
    if is_url and is_file: #URL과 파일이면
        source = check_file(source)  # 파일 확인
        
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 증가
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # create dir

    # Load model
    device = select_device(device) #장치 선택
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half) #모델 로드
    stride, names, pt = model.stride, model.names, model.pt #stride, names, pt를 model에서 가져옴
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Dataloader
    bs = 1  # batch_size
    if webcam: #웹캠이면
        view_img = check_imshow(warn=True) #imshow 가능한지 확인
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride) #스트림 로드
        bs = len(dataset) #batch_size
    elif screenshot: #스크린샷이면
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt) #스크린샷 로드
    else: #그 외
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride) #이미지 로드
    vid_path, vid_writer = [None] * bs, [None] * bs #비디오 경로, 비디오 작성자

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # run once
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device)) #seen, windows, dt를 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))로 초기화
    for path, im, im0s, vid_cap, s in dataset: #데이터셋에 대해 반복
        with dt[0]: #dt[0]에 대해
            im = torch.from_numpy(im).to(model.device) #im을 numpy에서 torch로 변환
            im = im.half() if model.fp16 else im.float()  # to FP16
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3: #im의 shape이 3이면
                im = im[None]  #  batch dim 확장
            if model.xml and im.shape[0] > 1: #xml이고 im의 shape이 1보다 크면
                ims = torch.chunk(im, im.shape[0], 0) #im을 im.shape[0]만큼 나눔

        # Inference
        with dt[1]: #dt[1]에 대해
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False # save_dir/Path(path).stem에 대한 시각화
            if model.xml and im.shape[0] > 1: #xml이고 im의 shape이 1보다 크면
                pred = None #pred를 None으로 초기화
                for image in ims: #ims에 대해 반복
                    if pred is None: #pred가 None이면
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0) #모델을 image에 대해 실행
                    else: #그 외
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0) #pred에 model(image, augment=augment, visualize=visualize)를 dim=0에 대해 연결
                pred = [pred, None] #pred를 [pred, None]으로 초기화
            else: #그 외
                pred = model(im, augment=augment, visualize=visualize) #모델을 im에 대해 실행
        # NMS
        with dt[2]: #dt[2]에 대해
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det) #비최대 억제

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"#csv_path를 save_dir/"predictions.csv"로 정의

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence): #write_to_csv 함수 정의
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence} #data를 {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}로 정의
            with open(csv_path, mode="a", newline="") as f: #csv_path를 열고
                writer = csv.DictWriter(f, fieldnames=data.keys()) #csv.DictWriter를 사용하여 writer를 정의
                if not csv_path.is_file():#csv_path가 파일이 아니면
                    writer.writeheader()#header를 작성
                writer.writerow(data)#data를 작성

        # Process predictions
        for i, det in enumerate(pred):  # detections per image #pred에 대해 반복
            seen += 1#seen을 1 증가
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count# p, im0, frame을 path[i], im0s[i].copy(), dataset.count로 정의
                s += f"{i}: "#s에 f"{i}: "를 추가
            else:#그 외
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0) #p, im0, frame을 path, im0s.copy(), getattr(dataset, "frame", 0)로 정의

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))#Annotator를 사용하여 annotator를 정의
            if len(det):#det의 길이가 0이 아니면
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()#박스 크기 조정

                # Print results
                for c in det[:, 5].unique():#det[:, 5]의 고유값에 대해 반복
                    n = (det[:, 5] == c).sum()  #클래스당 감지
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  #문자열에 추가

                # Write results
                for *xyxy, conf, cls in reversed(det): #det에 대해 반복
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"#label을 names[c]로 정의
                    confidence = float(conf)#confidence를 float(conf)로 정의
                    confidence_str = f"{confidence:.2f}"#confidence_str을 f"{confidence:.2f}"로 정의

                    if save_csv:# save_csv가 True이면
                        write_to_csv(p.name, label, confidence_str)#write_to_csv(p.name, label, confidence_str)를 실행

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # xywh 정규화
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f: #txt_path를 열고
                            f.write(("%g " * len(line)).rstrip() % line + "\n")#line을 작성

                    if save_img or save_crop or view_img: # save_img 또는 save_crop 또는 view_img이면
                        c = int(cls) # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}") #label을 정의
                        annotator.box_label(xyxy, label, color=colors(c, True))#box_label을 사용하여 annotator에 추가
                    if save_crop: # save_crop이면
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True) #save_one_box를 사용하여 imc에 저장

            # Stream results
            im0 = annotator.result() #annotator의 결과를 im0에 저장
            if view_img:#view_img이면
                if platform.system() == "Linux" and p not in windows:#platform.system()이 "Linux"이고 p가 windows에 없으면
                    windows.append(p) #windows에 p를 추가
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO) #윈도우 생성
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0]) #윈도우 크기 조정
                cv2.imshow(str(p), im0)#윈도우에 im0 표시
                cv2.waitKey(1) # 1ms 지연

            # Save results (image with detections)
            if save_img:#save_img이면
                if dataset.mode == "image": #이미지 모드이면
                    cv2.imwrite(save_path, im0)#im0을 save_path에 저장
                else: #그 외
                    if vid_path[i] != save_path: #vid_path[i]이 save_path와 같지 않으면
                        vid_path[i] = save_path #vid_path[i]를 save_path로 변경
                        if isinstance(vid_writer[i], cv2.VideoWriter):#vid_writer[i]가 cv2.VideoWriter이면
                            vid_writer[i].release() #해제
                        if vid_cap: #비디오 캡처가 있으면
                            fps = vid_cap.get(cv2.CAP_PROP_FPS) #fps를 가져옴
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #w를 가져옴
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))#h를 가져옴
                        else: #그 외
                            fps, w, h = 30, im0.shape[1], im0.shape[0] #fps, w, h를 30, im0.shape[1], im0.shape[0]으로 정의
                        save_path = str(Path(save_path).with_suffix(".mp4")) #save_path를 .mp4로 변경
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) #비디오 작성자 생성
                    vid_writer[i].write(im0) #im0을 vid_writer[i]에 작성

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms") #s, dt[1].dt * 1E3를 출력

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt) # tuple of average times #dt의 평균 시간의 튜플
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t) #속도 출력
    if save_txt or save_img: #save_txt 또는 save_img이면
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else "" #s를 정의
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}") #결과 저장
    if update:#update이면
        strip_optimizer(weights[0]) #모델 최적화 제거


def parse_opt(): #parse_opt 함수 정의
    parser = argparse.ArgumentParser() #argparse.ArgumentParser()를 사용하여 parser를 정의
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1 # expand
    print_args(vars(opt)) #인수를 출력
    return opt #opt를 반환


def main(opt): #main 함수 정의
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop")) #requirements.txt를 확인
    run(**vars(opt)) #run을 실행


if __name__ == "__main__": #이름이 "__main__"이면
    opt = parse_opt() #opt를 parse_opt()로 정의
    main(opt) #main을 실행
