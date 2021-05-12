import argparse

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from deep_sort import DeepSort

deepsort = DeepSort("deep_sort/deep/checkpoint/ckpt.t7")
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def bbox_rel(image_width, image_height, bbox_left, bbox_top, bbox_w, bbox_h):
    """" Calculates the relative bounding box from absolute pixel values. """
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def draw_boxes(img, bbox, identities,offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        label = "ID: " + '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.putText(img, label, (x1, y2 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

def detect(save_img=False):
    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, view_img, save_txt = opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, imgsz, quantized=opt.quantized, quantizer_output=opt.quantizer_output, a_bit=opt.a_bit,
                    w_bit=opt.w_bit,
                    FPGA=opt.FPGA)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'], strict=False)
    else:  # darknet format
        load_darknet_weights(model, weights)
    #################打印model_list
    '''AWEIGHT = torch.load(weights, map_location=device)['model']
    for k,v in AWEIGHT.items():
        print(k)'''

    # Eval mode
    model.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # _ = model(img.float()) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []
                dist = []
                for *xyxy, conf, cls in det:
                    img_h, img_w, _ = im0.shape  # get image shape
                    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
                    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
                    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
                    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, bbox_left, bbox_top, bbox_w, bbox_h)
                    distance = round(((1.7*1500*608)/(bbox_h*2.02))/1000)
                    dist.append(distance)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    t3 = torch_utils.time_synchronized()
                    outputs = deepsort.update((torch.Tensor(bbox_xywh)), (torch.Tensor(confs)) , im0)
                    t4 = torch_utils.time_synchronized()

                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        label = '%s %d %s' % (names[int(cls)],distance,"m")
                        draw_boxes(im0, bbox_xyxy,identities)
                        plot_one_box(xyxy,im0,label=label,color=colors[int(cls)])
                RS = 0
                closest = min(dist)
                if len(det) >= 3:
                    RS = 2
                if len(det) >= 4:
                    RS = 5
                if closest < 20:
                    ADD = 2.5
                    RS = RS + ADD
                cv2.putText(im0, "Risk Score: " + str(RS), (100,100), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2,2)[0]
                RS = 0
            # ds = t4 - t3
            # total = ds + (t2-t1)
            # #print("FPS: " + str(1/total))
            # print('%sDone. (%.3fs)' % (s, total))
            print('%sDone. (%.3fs)' % (s, t2-t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--quantized', type=int, default=0,
                        help='0:quantization way one Ternarized weight and 8bit activation')
    parser.add_argument('--a-bit', type=int, default=8,
                        help='a-bit')
    parser.add_argument('--w-bit', type=int, default=8,
                        help='w-bit')
    parser.add_argument('--FPGA', action='store_true', help='FPGA')
    parser.add_argument('--quantizer_output', type=bool, default=False, help='output')
    opt = parser.parse_args()
    opt.cfg = list(glob.iglob('./**/' + opt.cfg, recursive=True))[0]  # find file
    opt.names = list(glob.iglob('./**/' + opt.names, recursive=True))[0]  # find file
    print(opt)

    with torch.no_grad():
        detect()

if opt.quantizer_output == True:

    path = './quantier_output/q_bias_out'
    i = 1
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) == True:
            new_name = file.replace(file, "q_bias-modulelist_Conv2d_%d.txt" % (76 - i))
            os.rename(os.path.join(path, file), os.path.join(path, new_name))
            i += 1

    path = './quantier_output/q_weight_out'
    i = 1
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) == True:
            new_name = file.replace(file, "q_weight-modulelist_Conv2d_%d.txt" % (76 - i))
            os.rename(os.path.join(path, file), os.path.join(path, new_name))
            i += 1

    path = './quantier_output/q_activation_out'
    i = 1
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) == True:
            new_name = file.replace(file, "q_activation-modulelist_Conv2d_%d.txt" % (76 - i))
            os.rename(os.path.join(path, file), os.path.join(path, new_name))
            i += 1

    path = './quantier_output/b_scale_out'
    i = 1
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) == True:
            new_name = file.replace(file, "scale_bias-modulelist_Conv2d_%d.txt" % (76 - i))
            os.rename(os.path.join(path, file), os.path.join(path, new_name))
            i += 1

    path = './quantier_output/w_scale_out'
    i = 1
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) == True:
            new_name = file.replace(file, "scale_weight-modulelist_Conv2d_%d.txt" % (76 - i))
            os.rename(os.path.join(path, file), os.path.join(path, new_name))
            i += 1

    path = './quantier_output/a_scale_out'
    i = 1
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) == True:
            new_name = file.replace(file, "scale_activation-modulelist_Conv2d_%d.txt" % (76 - i))
            os.rename(os.path.join(path, file), os.path.join(path, new_name))
            i += 1

    #################输出每一层量化后的最大权值
    path = './quantier_output/q_weight_max'
    i = 1
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) == True:
            new_name = file.replace(file, "max_weight-modulelist_Conv2d_%d.txt" % (76 - i))
            os.rename(os.path.join(path, file), os.path.join(path, new_name))
            file = open(os.path.join(path, new_name), "r", encoding="utf-8")
            mystr1 = file.readline()  # 表示一次读取一行
            file_max = open('./quantier_output/q_weight_max/q_weight_max.txt', "a", encoding="utf-8")
            file_max.write(mystr1[:-1] + '\n')
            file_max.close()
            file.close()
            i += 1
    #################输出每一层量化后的最大激活
    path = './quantier_output/q_activation_max'
    i = 1
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) == True:
            new_name = file.replace(file, "max_activation-modulelist_Conv2d_%d.txt" % (76 - i))
            os.rename(os.path.join(path, file), os.path.join(path, new_name))
            # 合并最大值文档
            file = open(os.path.join(path, new_name), "r", encoding="utf-8", errors="ignore")
            mystr1 = file.readline()  # 表示一次读取一行
            file_max = open('./quantier_output/q_activation_max/q_activation_max.txt', "a", encoding="utf-8",
                            errors="ignore")
            file_max.write(mystr1[:-1] + '\n')
            file_max.close()
            file.close()
            i += 1

    ##########从这一行开始合并count文件
    path = './quantier_output/max_weight_count'
    i = 1
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) == True:
            new_name = file.replace(file, "max_weight_count-modulelist_Conv2d_%d.txt" % (76 - i))
            os.rename(os.path.join(path, file), os.path.join(path, new_name))
            file = open(os.path.join(path, new_name), "r", encoding="utf-8")
            mystr1 = file.readline()  # 表示一次读取一行
            file_max = open('./quantier_output/max_weight_count/max_weight_count.txt', "a", encoding="utf-8")
            file_max.write(mystr1[:-1] + '\n')
            file_max.close()
            file.close()
            i += 1

    path = './quantier_output/max_activation_count'
    i = 1
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) == True:
            new_name = file.replace(file, "max_activation_count-modulelist_Conv2d_%d.txt" % (76 - i))
            os.rename(os.path.join(path, file), os.path.join(path, new_name))
            file = open(os.path.join(path, new_name), "r", encoding="utf-8", errors="ignore")
            mystr1 = file.readline()  # 表示一次读取一行
            file_max = open('./quantier_output/max_activation_count/max_activation_count.txt', "a", encoding="utf-8",
                            errors="ignore")
            file_max.write(mystr1[:-1] + '\n')
            file_max.close()
            file.close()
            i += 1
