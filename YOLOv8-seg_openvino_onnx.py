import argparse
import time 
import cv2
import numpy as np
from openvino.runtime import Core  # pip install openvino -i  https://pypi.tuna.tsinghua.edu.cn/simple
import onnxruntime as ort  # 使用onnxruntime推理用上，pip install onnxruntime，默认安装CPU


# COCO默认的80类
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                      'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    
class OpenvinoInference(object):
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        ie = Core()
        self.model_onnx = ie.read_model(model=self.onnx_path)
        self.compiled_model_onnx = ie.compile_model(model=self.model_onnx, device_name="CPU")
        
    def predict(self, datas):
        # 注：self.compiled_model_onnx([datas])是一个字典，self.compiled_model_onnx.output(0)是字典键，第一种读取所有值方法(0.11s) 比 第二种按键取值的方法(0.20s) 耗时减半
        predict_data = list(self.compiled_model_onnx([datas]).values()) 
        # predict_data = [self.compiled_model_onnx([datas])[self.compiled_model_onnx.output(0)],
        #                  self.compiled_model_onnx([datas])[self.compiled_model_onnx.output(1)]]
        return predict_data
    

class YOLOv8_seg:
    """YOLOv8 segmentation model class for handling inference and visualization."""

    def __init__(self, onnx_model, imgsz=(640, 640), infer_tool='openvino'):
        """
        Initialization.

        Args:
            onnx_model (str): Path to the ONNX model.
        """
        self.infer_tool = infer_tool
        if self.infer_tool == 'openvino':
            # 构建openvino推理引擎
            self.openvino = OpenvinoInference(onnx_model)
            self.ndtype = np.single
        else:
            # 构建onnxruntime推理引擎
            self.ort_session = ort.InferenceSession(onnx_model,
                                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                                                if ort.get_device() == 'GPU' else ['CPUExecutionProvider'])

            # Numpy dtype: support both FP32 and FP16 onnx model
            self.ndtype = np.half if self.ort_session.get_inputs()[0].type == 'tensor(float16)' else np.single
       
        self.classes = CLASSES  # 加载模型类别
        self.model_height, self.model_width = imgsz[0], imgsz[1]  # 图像resize大小
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))  # 为每个类别生成调色板

    def __call__(self, im0, conf_threshold=0.4, iou_threshold=0.45, nm=32):
        """
        The whole pipeline: pre-process -> inference -> post-process.
        
        Args:
            im0 (Numpy.ndarray): original input image.
            conf_threshold (float): confidence threshold for filtering predictions.
            iou_threshold (float): iou threshold for NMS.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """
        # 前处理Pre-process
        t1 = time.time()
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)
        print('预处理时间：{:.3f}s'.format(time.time() - t1))
        
        # 推理 inference
        t2 = time.time()
        if self.infer_tool == 'openvino':
            preds = self.openvino.predict(im)
        else:
            preds = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: im})  # 与bbox区别，输出是个列表，[检测头的输出(1, 116, 8400), 分割头的输出(1, 32, 160, 160)]
        print('推理时间：{:.2f}s'.format(time.time() - t2))
        
        # 后处理Post-process
        t3 = time.time()
        boxes, segments, masks = self.postprocess(preds,
                                im0=im0,
                                ratio=ratio,
                                pad_w=pad_w,
                                pad_h=pad_h,
                                conf_threshold=conf_threshold,
                                iou_threshold=iou_threshold,
                                nm=nm
                                )
        print('后处理时间：{:.3f}s'.format(time.time() - t3))

        return boxes, segments, masks
        
    # 前处理，包括：resize, pad, HWC to CHW，BGR to RGB，归一化，增加维度CHW -> BCHW
    def preprocess(self, img):
        """
        Pre-processes the input image.

        Args:
            img (Numpy.ndarray): image about to be processed.

        Returns:
            img_process (Numpy.ndarray): image preprocessed for inference.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
        """
        # Resize and pad input image using letterbox() (Borrowed from Ultralytics)
        shape = img.shape[:2]  # original image shape
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # 填充

        # Transforms: HWC to CHW -> BGR to RGB -> div(255) -> contiguous -> add axis(optional)
        img = np.ascontiguousarray(np.einsum('HWC->CHW', img)[::-1], dtype=self.ndtype) / 255.0
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, ratio, (pad_w, pad_h)
    
    # 后处理，包括：阈值过滤+NMS+masks处理
    def postprocess(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
        """
        Post-process the prediction.

        Args:
            preds (Numpy.ndarray): predictions come from ort.session.run().
            im0 (Numpy.ndarray): [h, w, c] original input image.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
            conf_threshold (float): conf threshold.
            iou_threshold (float): iou threshold.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """
        x, protos = preds[0], preds[1]  # 与bbox区别：Two outputs: 检测头的输出(1, 116, 8400), 分割头的输出(1, 32, 160, 160)

        # Transpose the first output: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
        x = np.einsum('bcn->bnc', x)  # (1, 8400, 116)
   
        # Predictions filtering by conf-threshold，不包括后32维的向量（32维的向量可以看作是与每个检测框关联的分割 mask 的系数或权重）
        x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]

        # Create a new matrix which merge these(box, score, cls, nm) into one
        # For more details about `numpy.c_()`: https://numpy.org/doc/1.26/reference/generated/numpy.c_.html
        x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]

        # NMS filtering
        # 经过NMS后的值, np.array([[x, y, w, h, conf, cls, nm], ...]), shape=(-1, 4 + 1 + 1 + 32)
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]
        
        # 重新缩放边界框，为画图做准备
        if len(x) > 0:
            # Bounding boxes format change: cxcywh -> xyxy
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            # Bounding boxes boundary clamp
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])

            # 与bbox区别：增加masks处理
            # Process masks
            masks = self.process_mask(protos[0], x[:, 6:], x[:, :4], im0.shape)
            # Masks -> Segments(contours)
            segments = self.masks2segments(masks)
            
            return x[..., :6], segments, masks  # boxes, segments, masks
        else:
            return [], [], []

    @staticmethod
    def masks2segments(masks):
        """
        It takes a list of masks(n,h,w) and returns a list of segments(n,xy) (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L750)

        Args:
            masks (numpy.ndarray): the output of the model, which is a tensor of shape (batch_size, 160, 160).

        Returns:
            segments (List): list of segment masks.
        """
        segments = []
        for x in masks.astype('uint8'):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # CHAIN_APPROX_SIMPLE  该函数用于查找二值图像中的轮廓。
            if c:
                # 这段代码的目的是找到图像x中的最外层轮廓，并从中选择最长的轮廓，然后将其转换为NumPy数组的形式。
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # no segments found
            segments.append(c.astype('float32'))
        return segments

    
    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        """
        Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher quality
        but is slower. (Borrowed from https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L618)

        Args:
            protos (numpy.ndarray): [mask_dim, mask_h, mask_w].
            masks_in (numpy.ndarray): [n, mask_dim], n is number of masks after nms.
            bboxes (numpy.ndarray): bboxes re-scaled to original image shape.
            im0_shape (tuple): the size of the input image (h,w,c).

        Returns:
            (numpy.ndarray): The upsampled masks.
        """
        c, mh, mw = protos.shape
        masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
        masks = np.ascontiguousarray(masks)
        masks = self.scale_mask(masks, im0_shape)  # re-scale mask from P3 shape to original input image shape
        masks = np.einsum('HWN -> NHW', masks)  # HWN -> NHW
        masks = self.crop_mask(masks, bboxes)
        return np.greater(masks, 0.5)

    @staticmethod
    def scale_mask(masks, im0_shape, ratio_pad=None):
        """
        Takes a mask, and resizes it to the original image size. (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L305)

        Args:
            masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
            im0_shape (tuple): the original image shape.
            ratio_pad (tuple): the ratio of the padding to the original image.

        Returns:
            masks (np.ndarray): The masks that are being returned.
        """
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]),
                           interpolation=cv2.INTER_LINEAR)  # INTER_CUBIC would be better
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks
    
    @staticmethod
    def crop_mask(masks, boxes):
        """
        It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box. (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L599)

        Args:
            masks (Numpy.ndarray): [n, h, w] tensor of masks.
            boxes (Numpy.ndarray): [n, 4] tensor of bbox coordinates in relative point form.

        Returns:
            (Numpy.ndarray): The masks are being cropped to the bounding box.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
    
    # 绘框，与bbox区别：增加masks可视化
    def draw_and_visualize(self, im, bboxes, segments, vis=False, save=True):
        """
        Draw and visualize results.

        Args:
            im (np.ndarray): original image, shape [h, w, c].
            bboxes (numpy.ndarray): [n, 6], n is number of bboxes.
            segments (List): list of segment masks.
            vis (bool): imshow using OpenCV.
            save (bool): save image annotated.

        Returns:
            None
        """
        # Draw rectangles and polygons
        im_canvas = im.copy()
        # Draw rectangles 
        for (*box, conf, cls_), segment in zip(bboxes, segments):
            # draw contour and fill mask
            cv2.polylines(im, np.int32([segment]), True, (255, 255, 255), 2)  # white borderline
            cv2.fillPoly(im_canvas, np.int32([segment]), (255, 0, 0))

            # draw bbox rectangle
            cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          self.color_palette[int(cls_)], 1, cv2.LINE_AA)
            cv2.putText(im, f'{self.classes[int(cls_)]}: {conf:.3f}', (int(box[0]), int(box[1] - 9)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_palette[int(cls_)], 2, cv2.LINE_AA)

        # Mix image
        im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)

        # Show image
        if vis:
            cv2.imshow('demo', im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Save image
        if save:
            cv2.imwrite('demo.jpg', im)


if __name__ == '__main__':
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='weights\\yolov8s-seg.onnx', help='Path to ONNX model')
    parser.add_argument('--source', type=str, default=str('bus.jpg'), help='Path to input image')
    parser.add_argument('--imgsz', type=tuple, default=(640, 640), help='Image input size')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--infer_tool', type=str, default='openvino', choices=("openvino", "onnxruntime"), help='选择推理引擎')
    args = parser.parse_args()

    # Build model
    model = YOLOv8_seg(args.model, args.imgsz, args.infer_tool)

    # Read image by OpenCV
    img = cv2.imread(args.source)

    # Inference
    boxes, segments, _ = model(img, conf_threshold=args.conf, iou_threshold=args.iou)

    # Visualize, Draw bboxes and polygons
    if len(boxes) > 0:
        model.draw_and_visualize(img, boxes, segments, vis=False, save=True)

