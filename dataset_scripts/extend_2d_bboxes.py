"""
This script will extend the 3D labels including the 2D bboxes only for cam front.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import argparse

def get_3d_box(w, h, l, x, y, z, yaw):
    # 8 vértices en coordenadas locales del objeto (x delante, y izquierda, z arriba)
    x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [ w/2,  w/2,  w/2,  w/2, -w/2, -w/2, -w/2, -w/2]
    z_corners = [ h, 0, 0, h, h, 0, 0, h]  # z=0 en la base y z=h en el techo

    corners = np.array([x_corners, y_corners, z_corners])  # shape (3, 8)

    # Rotación alrededor del eje Z (yaw en radianes)
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,           0,            1]
    ])

    # Aplicar rotación y traslación
    corners_3d = R @ corners
    corners_3d += np.array([[x], [y], [z]])
    return corners_3d.T  # shape (8, 3)

def transform_to_cam(corners_ego, ego2cam):
    corners_hom = np.concatenate([corners_ego, np.ones((8,1))], axis=1).T  # shape (4, 8)
    corners_cam = ego2cam @ corners_hom  # shape (4, 8)
    return corners_cam[:3].T  # shape (8, 3)

def project_to_image(corners_cam, cam2img):
    corners_hom = np.concatenate([corners_cam, np.ones((8,1))], axis=1).T  # shape (4, 8)
    img_points = cam2img @ corners_hom  # shape (3, 8)
    img_points = img_points[:2] / img_points[2]
    return img_points.T  # shape (8, 2)

def draw_box(image, img_pts, color=(0, 255, 0), thickness=2):
    img_pts = img_pts.astype(int)

    # Conecta los vértices para formar las aristas
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # base
        (4, 5), (5, 6), (6, 7), (7, 4),  # techo
        (0, 4), (1, 5), (2, 6), (3, 7)   # verticales
    ]
    
    for i, j in connections:
        pt1, pt2 = tuple(img_pts[i]), tuple(img_pts[j])
        cv2.line(image, pt1, pt2, color, thickness)
    
    return image

def get_2d_bbox_and_depth(img_pts, corners_cam):
    u_min = int(np.min(img_pts[:, 0]))
    v_min = int(np.min(img_pts[:, 1]))
    u_max = int(np.max(img_pts[:, 0]))
    v_max = int(np.max(img_pts[:, 1]))
    bbox = (u_min, v_min, u_max, v_max)

    # Profundidad media (Z en coordenadas cámara)
    depth = np.mean(corners_cam[:, 2])
    return bbox, depth

def draw_bbox_2d(image, bbox, color=(0, 255, 0), thickness=2):
    u_min, v_min, u_max, v_max = bbox
    cv2.rectangle(image, (u_min, v_min), (u_max, v_max), color, thickness)
    return image

def get_cam_front_sample(label_path, calib_path, img_path, img_inst_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
        labels = [line.strip() for line in lines if line.strip()]

    with open(calib_path, 'r') as f:
        lines = f.readlines()
        calib = [line.strip() for line in lines if line.strip()]
        cam_front_k = []
        lidar2cam_front = []
        for line in calib:
            if line.startswith('CAM_FRONT_K:'):
                cam_front_k = [float(x) for x in line.split()[1:]]
                cam_front_k = np.array(cam_front_k).reshape(4, 4)
            elif line.startswith('LIDAR2CAM_FRONT:'):
                lidar2cam_front = [float(x) for x in line.split()[1:]]
                lidar2cam_front = np.array(lidar2cam_front).reshape(4, 4)
            elif line.startswith('LIDAR2EGO:'):
                lidar2ego = [float(x) for x in line.split()[1:]]
                lidar2ego = np.array(lidar2ego).reshape(4, 4)

    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_inst = cv2.imread(img_inst_path)
    img_inst = cv2.cvtColor(img_inst, cv2.COLOR_BGR2RGB)

    return img, img_inst, labels, cam_front_k, lidar2cam_front, lidar2ego

def compute_iou(boxA, boxB):
    # Coordenadas del área de intersección
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Área de intersección
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    # Área de cada caja
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # IoU

    if boxAArea == 0 or boxBArea == 0:
        return 0.0

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# filter occlusions
def filter_occluded_boxes(bboxes, depths, items, image_shape, threshold=0.8):
    """
    bboxes: lista de (u_min, v_min, u_max, v_max)
    depths: lista de profundidades (mismo orden que bboxes)
    items: lista de ids de los objetos (mismo orden que bboxes)
    image_shape: (H, W) de la imagen
    threshold: porcentaje de oclusión permitido (0.8 = 80%)
    """

    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)

    # Ordenar por profundidad (cerca a lejos)
    sorted_idx = np.argsort(depths)
    visible_boxes = []
    visible_items = []

    for idx in sorted_idx:
        u_min, v_min, u_max, v_max = bboxes[idx]
        u_min, v_min = max(0, u_min), max(0, v_min)
        u_max, v_max = min(W - 1, u_max), min(H - 1, v_max)

        # Calcular área total de la caja
        area_total = (u_max - u_min + 1) * (v_max - v_min + 1)
        if area_total <= 0:
            continue

        # Calcular área ya ocupada (ocluida) en el mask
        mask_crop = mask[v_min:v_max + 1, u_min:u_max + 1]
        area_occluded = np.count_nonzero(mask_crop)

        oclusion_ratio = area_occluded / area_total

        if oclusion_ratio <= threshold:
            visible_boxes.append(idx)
            visible_items.append(items[idx])
            # Marcar esta caja como visible
            mask[v_min:v_max + 1, u_min:u_max + 1] = 1

    return visible_boxes, visible_items

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extend 2D bboxes from 3D labels')
    parser.add_argument("--split", "-s", type=str, required=True, help="Path to the split")
    args = parser.parse_args()

    dataset_path = "/home/nupdm/Datasets/nuPDM/" + args.split

    # list all routes
    route_names = []
    for route in sorted(os.listdir(dataset_path)):
        if "route" not in route:
            continue
        route_number = route.split("_")[1]
        route_name = os.path.join(dataset_path, route)
        route_names.append(route_name)


    for route in route_names:
        DATAROOT = route
        LABELS_PATH = os.path.join(DATAROOT, "labels")
        CALIB_PATH = os.path.join(DATAROOT, "calib")
        CAM_FRONT_PATH = os.path.join(DATAROOT, "CAM_FRONT")
        CAM_FRONT_INST_PATH = os.path.join(DATAROOT, "CAM_FRONT_INST")
        print("DATAROOT: ", DATAROOT)

        n_files = len([name for name in os.listdir(LABELS_PATH) if os.path.isfile(os.path.join(LABELS_PATH, name))])
        print("Number of files in boxes folder: ", n_files)

        for ITEM in range(n_files):
            ITEM = str(ITEM).zfill(4)

            label_path = os.path.join(LABELS_PATH, ITEM + ".txt")
            calib_path = os.path.join(CALIB_PATH, ITEM + ".txt")
            img_path = os.path.join(CAM_FRONT_PATH, ITEM + ".jpg")
            img_inst_path = os.path.join(CAM_FRONT_INST_PATH, ITEM + ".png")
            print("Processing: ", label_path)

            img, img_inst, labels, cam_front_k, lidar2cam_front, lidar2ego = get_cam_front_sample(label_path, calib_path, img_path, img_inst_path)

            ego2cam = lidar2cam_front @ np.linalg.inv(lidar2ego)  # Transformación de ego a cámara
            cam2img = cam_front_k[:3, :]

            # Getting 2D bboxes from 3D labels
            objects = []
            corners_3d_list = []
            bboxes_2d_list = []
            depths_list = []
            item_id_list = []

            for i, line in enumerate(labels):
                label = line.split()
                if len(label) == 9: # if it doesnot have 9, it already has 2d label
                    cls, w, h, l, x, y, z, yaw, _ = label
                else:
                    cls, w, h, l, x, y, z, yaw, _ = label[:9]

                if float(x) < 0.5:
                    continue

                x = float(x)
                y = float(y)
                z = float(z)
                h = float(h)
                w = float(w)
                l = float(l)
                yaw = float(yaw)

                corners_3d = get_3d_box(w, h, l, x, y, z, yaw)
                corners_3d_list.append(corners_3d)
                corners_cam = transform_to_cam(corners_3d, ego2cam)
                corners_2d = project_to_image(corners_cam, cam2img)

                # filter corners_2d so that if x1 < 0 or x2 > img.shape[1] or y1 < 0 or y2 > img.shape[0] then set thos values to 0 or img.shape[1] or img.shape[0]
                corners_2d[:, 0] = np.clip(corners_2d[:, 0], 0, img.shape[1] - 1)
                corners_2d[:, 1] = np.clip(corners_2d[:, 1], 0, img.shape[0] - 1)

                bboxes_2d, depths = get_2d_bbox_and_depth(corners_2d, corners_cam)
                bboxes_2d_list.append(bboxes_2d)
                depths_list.append(depths)
                item_id_list.append(i)

                # Draw 2D boxes
                # img = draw_bbox_2d(img, bboxes_2d, color=(255, 0, 0), thickness=2)

            # Getting 2D bboxes from 2D CAM_FRONT_INST
            pixels = img_inst.reshape(-1, 3)
            unique_ids = np.unique(pixels, axis=0)
            # get ids for classes 7, 8, 12, 14, 15, 16, 18, 19 (check semantic lidar docs)
            unique_ids = unique_ids[
                (unique_ids[:, 0] == 14)
                | (unique_ids[:, 0] == 12)
                | (unique_ids[:, 0] == 15)
                | (unique_ids[:, 0] == 16)
                | (unique_ids[:, 0] == 18)
                | (unique_ids[:, 0] == 19)
                | (unique_ids[:, 0] == 7)
                | (unique_ids[:, 0] == 8)
                | (unique_ids[:, 0] == 21) # static traffic warning)
            ]

            bboxes_2d_list_inst = []
            for id in unique_ids:
                positions = np.all(img_inst == id, -1)
                positions = np.argwhere(positions)

                if positions.size > 0:
                    min_y = np.min(positions[:, 0])
                    max_y = np.max(positions[:, 0])
                    min_x = np.min(positions[:, 1])
                    max_x = np.max(positions[:, 1])
                    bboxes_2d_list_inst.append([min_x, min_y, max_x, max_y])

                # draw a rectangle around the positions
                cv2.rectangle(
                    img_inst,
                    (min_x, min_y),
                    (max_x, max_y),
                    (255, 0, 0),
                    2,
                )

            # Get the associations between both
            iou_threshold = 0.2
            bboxes_2d_list_associated = []
            depths_associated = []
            item_id_list_associated = []
            if len(bboxes_2d_list) > len(bboxes_2d_list_inst):
                for i, inst in enumerate(bboxes_2d_list_inst):
                    best_iou = 0
                    best_match = -1
                    for j, geom_2d in enumerate(bboxes_2d_list):
                        iou = compute_iou(inst, geom_2d)
                        if iou > best_iou:
                            best_iou = iou
                            best_match = j
                    if best_iou >= iou_threshold:
                        bboxes_2d_list_associated.append(bboxes_2d_list_inst[i]) # always keep the instance idx
                        depths_associated.append(depths_list[best_match])
                        item_id_list_associated.append(item_id_list[best_match])
            else:
                for i, geom_2d in enumerate(bboxes_2d_list):
                    best_iou = 0
                    best_match = -1
                    for j, inst in enumerate(bboxes_2d_list_inst):
                        iou = compute_iou(geom_2d, inst)
                        if iou > best_iou:
                            best_iou = iou
                            best_match = j
                    if best_iou >= iou_threshold:
                        bboxes_2d_list_associated.append(bboxes_2d_list_inst[best_match]) # always keep the instance idx
                        depths_associated.append(depths_list[i])
                        item_id_list_associated.append(item_id_list[i])


            visible_boxes, visible_items = filter_occluded_boxes(bboxes_2d_list_associated, depths_associated, item_id_list_associated, img.shape[:2], threshold=0.8)
            
            final_2d_bboxes = {
                "bbox_2d_list": bboxes_2d_list_associated,
                "label_related_id": item_id_list_associated,
            }

            new_labels = []
            for i, bbox in enumerate(final_2d_bboxes["bbox_2d_list"]):
                u_min, v_min, u_max, v_max = bbox
                label = labels[final_2d_bboxes["label_related_id"][i]]
                new_label = label.split()
                # add u_min, v_min, u_max, v_max to the label
                new_label.append(str(u_min))
                new_label.append(str(v_min))
                new_label.append(str(u_max))
                new_label.append(str(v_max))
                new_labels.append(new_label)

            # open label_path and append the new labels, removing those in label_related_id
            with open(label_path, 'r') as f:
                lines = f.readlines()
                labels = [line.strip() for line in lines if line.strip()]
                # maybe close?
            f.close()

            with open(label_path, 'w') as f:
                for i, line in enumerate(labels):
                    if i in final_2d_bboxes["label_related_id"]:
                        continue
                    f.write(line + "\n")
                    # write the new labels
                for new_label in new_labels:
                    # convert new label from a list of str to a single str
                    f.write(" ".join(map(str, new_label)) + "\n")
                    # print("new_label: ", new_label)
            
            # Draw 2D boxes
            for idx in visible_boxes:
                u_min, v_min, u_max, v_max = bboxes_2d_list_associated[idx]
                img = cv2.rectangle(img, (u_min, v_min), (u_max, v_max), (0, 255, 0), 2)

            # my 2d bbox is:

            # show during 1 second and keep going

            # cv2.imshow(ITEM, img)
            # cv2.waitKey(2000)
            # cv2.destroyAllWindows()
            # save the image
            # img_path = os.path.join(DATAROOT, "2d_gt", ITEM + ".jpg")
            # os.makedirs(os.path.dirname(img_path), exist_ok=True)
            # cv2.imwrite(img_path, img)








