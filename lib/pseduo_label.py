import os
from PIL import Image
from lib.detect import detect
import numpy as np
from tqdm import tqdm
import cv2
from lib.detect import resize_img,load_pil,adjust_ratio,restore_polys,plot_boxes
import torch
import lanms
import shapely
from shapely.geometry import Polygon
from lib.swt import SWT


def get_negative_sample(score_map,negative_thresh=0.5):
    '''Get the third negative samples
       reduce FN
    	Input:
    		 score map:  <numpy.ndarray, (m,n)>
    	Output:
    		negative map: <numpy.ndarray, (m,n)>
    '''
    score_map = score_map[0, :, :]
    negative_map = np.zeros(score_map.shape)
    postive_score = (score_map>negative_thresh)*1.0
    # 膨胀
    kernel = np.ones((4, 4), np.uint8)
    postive_score = cv2.dilate(postive_score, kernel, iterations=1)

    xy_negative = np.argwhere(score_map*(1-postive_score)) 
    xy_negative = xy_negative[np.argsort(xy_negative[:, 0])]
    valid_score = score_map[xy_negative[:, 0], xy_negative[:, 1]]  # 5 x n
    xy_negative = xy_negative[np.argsort(valid_score)]    
    # negative_number = xy_negative.shape[0]                
    xy_negative = xy_negative[:int(xy_negative.shape[0] / 3), :].T #取前1/3的值作为负样本进行监督
    negative_map[xy_negative[0], xy_negative[1]] = 1

    return negative_map

def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2):
	'''get boxes from feature map
	Input:
		score       : score map from model <numpy.ndarray, (1,row,col)>
		geo         : geo map from model <numpy.ndarray, (5,row,col)>
		score_thresh: threshold to segment score map
		nms_thresh  : threshold in nms
	Output:
		boxes       : final polys <numpy.ndarray, (n,9)>
	'''
	score = score[0,:,:]
	xy_text = np.argwhere(score > score_thresh) # n x 2, format is [r, c]
	if xy_text.size == 0:
		return None,None

	xy_text = xy_text[np.argsort(xy_text[:, 0])]
	valid_pos = xy_text[:, ::-1].copy() # n x 2, [x, y]
	valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]] # 5 x n
	polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape)
	if polys_restored.size == 0:
		return None,None
	boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
	boxes[:, :8] = polys_restored
	boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
	after_boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
	return boxes.astype('float32'),after_boxes


def detection_pseudo(img, model, device):
    '''detect text regions of img using model
	Input:
		img   : PIL Image
		model : detection model
		device: gpu if gpu is available
	Output:
		detected polys
	'''

    img, ratio_h, ratio_w = resize_img(img)
    with torch.no_grad():
        (score, geo),domain_class = model(load_pil(img).to(device))
    befor_boxes, after_NMS_box = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())

    after_NMS_box = adjust_ratio(after_NMS_box, ratio_w, ratio_h)

    return after_NMS_box, score.squeeze(0).cpu().numpy(), ratio_w, ratio_h

def Supporting_Region_based_Reliable_Score_average(NMS_box,score, ratio_w, ratio_h,n_thresh = 0.7,thresh_score=0.8):
    '''Get the score of positive samples
           reduce FP
        	Input:
        		 score map:  <numpy.ndarray, (m,n)>
        	Output:
        		negative map: <numpy.ndarray, (m,n)>
        '''
    reliable_score = []

    score = cv2.resize(score[0],(ratio_w,ratio_h))
    for i, box in enumerate(NMS_box):
        poly = np.around(box[:8].reshape((4, 2))).astype(np.int32)
        reliable_map = np.zeros((ratio_h,ratio_w))
        cv2.fillPoly(reliable_map, [poly], 1)

        reliable_map = ((reliable_map*score) > n_thresh).astype(np.uint8) * 1
        confidence = (reliable_map*score).sum() / (reliable_map.sum()+1)

        if confidence>thresh_score:
            reliable_score.append(box)
    return np.array(reliable_score)

def Supporting_Region_based_Reliable_Score_Polygon(befor_boxes, NMS_box,n_thresh = 0.8):
    '''Get the score of positive samples
       reduce FP
    	Input:
    		 score map:  <numpy.ndarray, (m,n)>
    	Output:
    		negative map: <numpy.ndarray, (m,n)>
    '''
    final_box = []
    for i, box_1 in enumerate(NMS_box):
        postive_proposal = 0
        postive_proposal_iou = 0
        box_1_coordinate = np.around(box_1[:8].reshape((4, 2))).astype(np.int32)
        for box_2 in befor_boxes:
            box_2 = np.around(box_2[:8].reshape((4, 2))).astype(np.int32)
            iou = get_iou(box_1_coordinate,box_2)
            if iou>n_thresh:
                postive_proposal += 1
                postive_proposal_iou += iou
        if postive_proposal!=0:
            postive_proposal_iou = postive_proposal_iou/postive_proposal
        # 挑选iou大于SRRS_thred的proposal box进行得分统计， 因为更大面积的text必然有更多的框，这是不公平的对于score的计算
        # n: bigger area have more box and smaller area have less box
        if postive_proposal_iou>0.81:
            final_box.append(box_1)
    return np.array(final_box)

def Supporting_Region_based_Reliable_Score(befor_boxes, NMS_box,ratio_w, ratio_h,n_thresh = 0.9):
    '''Get the score of positive samples
       reduce FP
    	Input:
    		 score map:  <numpy.ndarray, (m,n)>
    	Output:
    		negative map: <numpy.ndarray, (m,n)>
    '''
    before_NMS = []
    ious = []
    final_box = []
    # box  to  map
    for box in befor_boxes:
        befor_map = np.zeros((ratio_h,ratio_w))
        coordinate = np.around(box[:8].reshape((4, 2))).astype(np.int32)
        cv2.fillPoly(befor_map,[coordinate],1)
        before_NMS.append(befor_map.flatten())
    before_NMS = np.array(before_NMS)
    for i, box in enumerate(NMS_box):
        poly = np.around(box[:8].reshape((4, 2))).astype(np.int32)
        after_map = np.zeros((ratio_h, ratio_w))
        cv2.fillPoly(after_map, [poly], 1)
        after_map = after_map.flatten()
        # 1对多进行IOU计算
        intersection = (after_map * before_NMS).sum(axis=1)
        union = after_map.sum() + before_NMS.sum(axis=1) - intersection

        SRRS = ((intersection + 1) / (union + 1))

        # 挑选iou大于SRRS_thred的proposal box进行得分统计， 因为更大面积的text必然有更多的框，这是不公平的对于score的计算
        postive_proposal_iou = (SRRS>n_thresh).astype(np.uint8) * 1
        n = postive_proposal_iou.sum()
        SRRS = ((SRRS*postive_proposal_iou).sum())/(n+1)
        # n: bigger area have more box and smaller area have less box
        ious.append(SRRS)
        if  SRRS>0.5:
            final_box.append(box)
    return np.array(final_box)

def get_SWT(image,boxes):
    stw = SWT(image, boxes)
    final_boxes = stw.detect()
    return final_boxes

def generate_pseduo(model, input_path,output_path,negative_path,device):
    '''generate pseudo label
    	Input:
    	Output:
    '''
    model.eval()
    image_list = os.listdir(input_path)
    print("            ----------------------------------------------------------------")
    print("                   Start generate pseudo-label of target domain")
    print("            ----------------------------------------------------------------")
    for one_image in tqdm(image_list):

        # print(one_image)
        image_path = os.path.join(input_path, one_image)
        img = Image.open(image_path)

        filename, file_ext = os.path.splitext(os.path.basename(one_image))
        res_file = output_path + filename + '.txt'
        negative_file = negative_path + filename + '.png'

        after_NMS_box,score, ratio_w, ratio_h = detection_pseudo(img, model, device)

        # 生产negative map
        negative_map = get_negative_sample(score)
        w, h = img.size
        negative_map = cv2.resize(negative_map, (w, h))

        with open(res_file, 'w') as f:
            if after_NMS_box is None:
                cv2.imwrite(negative_file, negative_map * 0)
                continue
            for i, box in enumerate(after_NMS_box):
                poly = np.array(box).astype(np.int32)
                points = np.reshape(poly, -1)
                strResult = ','.join(
                    [str(points[0]), str(points[1]), str(points[2]), str(points[3]), str(points[4]), str(points[5]),
                     str(points[6]), str(points[7])]) + '\r\n'
                f.write(strResult)

                # print(points[0], points[1], points[2], points[3], points[4], points[5], points[6], points[7])
                #排除positive map部分
                # cv2.fillPoly(negative_map, np.array(
                #     [[[points[0], points[1]], [points[2], points[3]], [points[4], points[5]], [points[6], points[7]]]]),(0))

        cv2.imwrite(negative_file, negative_map)


def get_iou(a,b):
	'''
	:param a: box a [x0,y0,x1,y1,x2,y2,x3,y3]
	:param b: box b [x0,y0,x1,y1,x2,y2,x3,y3]
	:return: iou of bbox a and bbox b
	'''
	# a = a.reshape(4, 2)
	poly1 = Polygon(a).convex_hull
	# b = b.reshape(4, 2)
	poly2 = Polygon(b).convex_hull
	if not poly1.intersects(poly2):  # 如果两四边形不相交
		iou = 0
	else:
		try:
			inter_area = poly1.intersection(poly2).area  # 相交面积
			union_area = poly1.area + poly2.area - inter_area
			if union_area == 0:
				iou = 0
			else:
				iou = inter_area / union_area
		except shapely.geos.TopologicalError:
			print('shapely.geos.TopologicalError occured, iou set to 0')
			iou = 0
	return iou