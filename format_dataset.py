import os
import json
import gzip
import random

import numpy as np
import laspy

from pdm_lite.team_code.config_nusc import GlobalConfigNusc

# TODO: Bug .bin points, .txt labels, 

def create_transformation_matrix(rotation_matrix, translation_vector):
    """
    Crea una matriz de transformación homogénea 4x4 a partir de
    una matriz de rotación 3x3 y un vector de traslación 3x1
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation_vector
    return transform

def check_collisions(route_folder: str):
    RESULTS_PATH = os.path.join(route_folder, "results.json.gz")
    MEASUREMENTS_FOLDER = os.path.join(route_folder, "measurements")

    with gzip.open(RESULTS_PATH, 'rt', encoding='utf-8') as f:
        # load json
        results = json.load(f)

    collisions = {
        "agent_collided": [],
        "id_collided": [],
        "x_collided": [],
        "y_collided": [],
    }

    for key in results['infractions'].keys():
        if 'collision' in key:
            collision = results['infractions'][key]
            if len(collision) > 0:
                for coll in collision:
                    # get id of collided element and position
                    # split by " "
                    coll = coll.split(" ")
                    for element in coll:
                        if "type=" in element:
                            element = element.split("=")[1]
                            collisions["agent_collided"].append(element)
                        if "id=" in element:
                            element = element.split("=")[1]
                            collisions["id_collided"].append(element)
                        if "x=" in element:
                            element = element.split("=")[1]
                            element = element[:-1]
                            collisions["x_collided"].append(element)
                        if "y=" in element:
                            element = element.split("=")[1]
                            element = element[:-1]
                            collisions["y_collided"].append(element)
                        

    files = os.listdir(MEASUREMENTS_FOLDER)
    # # files are in format: 0000.json.gz
    files = sorted(files)

    pos_coll = [(float(x), float(y)) for x, y in zip(collisions["x_collided"], collisions["y_collided"])]

    thres = 5
    invalid_files = []

    for file in files: 
        # check if file is json.gz
        if file.endswith(".json.gz"):
            with gzip.open(os.path.join(MEASUREMENTS_FOLDER, file), 'rt', encoding='utf-8') as f:
                # load json
                data = json.load(f)
                pos_global = data['pos_global']            

                for coll in pos_coll:
                    if -thres < pos_global[0] - coll[0] < thres and -thres < pos_global[1] - coll[1] < thres:
                        invalid_files.append(file.split(".json.gz")[0])

    # save invalid files in a text file
    with open(os.path.join(route_folder, "invalid_files.txt"), "w") as f:
        for file in invalid_files:
            f.write(file + "\n")

    return invalid_files

if __name__ == "__main__":
    # parse args
    import argparse
    parser = argparse.ArgumentParser(description="Format dataset")
    parser.add_argument("--split", "-s", type=str, default="routes_training_mini", help="Split Type")
    args = parser.parse_args()

    config = GlobalConfigNusc()
    # config.lidar_pos = [0.0, 0.0, 0.0]

    DATASET_PATH = "/home/nupdm/Datasets/nuPDM/" + args.split

    # list all routes
    route_names = []
    for route in sorted(os.listdir(DATASET_PATH)):
        if "route" not in route:
            continue
        route_number = route.split("_")[1]
        route_name = os.path.join(DATASET_PATH, route)
        route_names.append(route_name)


    for route in route_names:
        DATAROOT = route
        BOXES_PATH = os.path.join(DATAROOT, "boxes")
        MEASUREMENTS_PATH = os.path.join(DATAROOT, "measurements")
        print("DATAROOT: ", DATAROOT)

        # read number of files in the folders
        n_files = len([name for name in os.listdir(BOXES_PATH) if os.path.isfile(os.path.join(BOXES_PATH, name))])
        print("Number of files in boxes folder: ", n_files)

        for ITEM in range(n_files):
            ITEM = str(ITEM).zfill(4)
            # load measurements
            with gzip.open(os.path.join(MEASUREMENTS_PATH, ITEM + ".json.gz"), "rb") as f:
                measurements = json.load(f)

            # load boxes
            with gzip.open(os.path.join(BOXES_PATH, ITEM + ".json.gz"), "rb") as f:
                boxes = json.load(f)

            P_CAM_FRONT = config.camera_front_K
            P_CAM_FRONT_LEFT = config.camera_front_left_K
            P_CAM_FRONT_RIGHT = config.camera_front_right_K
            P_CAM_BACK = config.camera_back_K
            P_CAM_BACK_LEFT = config.camera_back_left_K
            P_CAM_BACK_RIGHT = config.camera_back_right_K

            # ------------------------------- Transformation matrixes -------------------------------------
            ego2lidar = config.ego2lidar
            ego2cam_front = config.ego2camera_front
            ego2cam_front_left = config.ego2camera_front_left
            ego2cam_front_right = config.ego2camera_front_right
            ego2cam_back = config.ego2camera_back
            ego2cam_back_left = config.ego2camera_back_left
            ego2cam_back_right = config.ego2camera_back_right
            ego2radar_front = config.ego2radar_front
            ego2radar_front_left = config.ego2radar_front_left
            ego2radar_front_right = config.ego2radar_front_right
            ego2radar_back_left = config.ego2radar_back_left
            ego2radar_back_right = config.ego2radar_back_right

            lidar2ego = np.linalg.inv(ego2lidar)
            
            lidar2cam_front = ego2cam_front @ lidar2ego
            lidar2cam_front_left = ego2cam_front_left @ lidar2ego
            lidar2cam_front_right = ego2cam_front_right @ lidar2ego
            lidar2cam_back = ego2cam_back @ lidar2ego
            lidar2cam_back_left = ego2cam_back_left @ lidar2ego
            lidar2cam_back_right = ego2cam_back_right @ lidar2ego
            lidar2radar_front = ego2radar_front @ lidar2ego
            lidar2radar_front_left = ego2radar_front_left @ lidar2ego
            lidar2radar_front_right = ego2radar_front_right @ lidar2ego
            lidar2radar_back_left = ego2radar_back_left @ lidar2ego
            lidar2radar_back_right = ego2radar_back_right @ lidar2ego
            # ---------------------------------------------------------------------------------------------

            ego_matrix = np.array(measurements["ego_matrix"])

            # create the directory if it does not exist
            os.makedirs(os.path.join(DATAROOT, "calib"), exist_ok=True)

            with open(os.path.join(DATAROOT, "calib", ITEM + ".txt"), "w") as f:
                f.write("CAM_FRONT_K: " + " ".join(map(str, P_CAM_FRONT.flatten())) + "\n")
                f.write("CAM_FRONT_LEFT_K: " + " ".join(map(str, P_CAM_FRONT_LEFT.flatten())) + "\n")
                f.write("CAM_FRONT_RIGHT_K: " + " ".join(map(str, P_CAM_FRONT_RIGHT.flatten())) + "\n")
                f.write("CAM_BACK_K: " + " ".join(map(str, P_CAM_BACK.flatten())) + "\n")
                f.write("CAM_BACK_LEFT_K: " + " ".join(map(str, P_CAM_BACK_LEFT.flatten())) + "\n")
                f.write("CAM_BACK_RIGHT_K: " + " ".join(map(str, P_CAM_BACK_RIGHT.flatten())) + "\n")
                f.write("LIDAR2CAM_FRONT: " + " ".join(map(str, lidar2cam_front.flatten())) + "\n")
                f.write("LIDAR2CAM_FRONT_LEFT: " + " ".join(map(str, lidar2cam_front_left.flatten())) + "\n")
                f.write("LIDAR2CAM_FRONT_RIGHT: " + " ".join(map(str, lidar2cam_front_right.flatten())) + "\n")
                f.write("LIDAR2CAM_BACK: " + " ".join(map(str, lidar2cam_back.flatten())) + "\n")
                f.write("LIDAR2CAM_BACK_LEFT: " + " ".join(map(str, lidar2cam_back_left.flatten())) + "\n")
                f.write("LIDAR2CAM_BACK_RIGHT: " + " ".join(map(str, lidar2cam_back_right.flatten())) + "\n")
                f.write("LIDAR2RADAR_FRONT: " + " ".join(map(str, lidar2radar_front.flatten())) + "\n")
                f.write("LIDAR2RADAR_FRONT_LEFT: " + " ".join(map(str, lidar2radar_front_left.flatten())) + "\n")
                f.write("LIDAR2RADAR_FRONT_RIGHT: " + " ".join(map(str, lidar2radar_front_right.flatten())) + "\n")
                f.write("LIDAR2RADAR_BACK_LEFT: " + " ".join(map(str, lidar2radar_back_left.flatten())) + "\n")
                f.write("LIDAR2RADAR_BACK_RIGHT: " + " ".join(map(str, lidar2radar_back_right.flatten())) + "\n")
                f.write("LIDAR2EGO: " + " ".join(map(str, lidar2ego.flatten())) + "\n")
                f.write("EGO_MATRIX: " + " ".join(map(str, ego_matrix.flatten())) + "\n")

            items = []

            # now go for the labels with format [<object_type> <truncation> <occlusion> <alpha> <left> <top> <right> <bottom> <height> <width> <length> <x> <y> <z> <rotation_y> <num_points>]
            # x, y, z are the coordinates of the center of the object
            # dx, dy, dz are the dimensions of the object
            # yaw is the rotation of the object
            # category is the category of the object

            for i, box in enumerate(boxes):
                if box["class"] == "car" or box["class"] == "static_car": # coche
                    # CARLA base types are car, truck, van, bus, bicycle
                    if ("base_type" in box):
                        base_type = box["base_type"]
                    else:
                        base_type = "car"
                    items.append([base_type, 
                            np.array(box["extent"][1])*2,   # w
                            np.array(box["extent"][2])*2,   # h
                            np.array(box["extent"][0])*2,   # l
                            box["position"][0],             # x
                            -box["position"][1],            # -y
                            box["position"][2],             # z
                            -float(box["yaw"]),             # -yaw
                            box["num_points"]])
                elif box["class"] == "walker":
                    items.append(["walker", 
                                # np.array(box["extent"][1])*2, 
                                # np.array(box["extent"][2])*2, 
                                0.5,
                                np.array(box["extent"][2])*2,
                                0.5,
                                box["position"][0],             
                                -box["position"][1], 
                                # box["position"][2], 
                                0,
                                -float(box["yaw"]), 
                                box["num_points"]])
                elif box["class"] == "traffic_light_vqa" or box["class"] == "traffic_light":
                        # if not box["affects_ego"]:
                        #     continue
                        if box["state"] == "Red":
                            state = 0
                        elif box["state"] == "Yellow":
                            state = 1
                        elif box["state"] == "Green":
                            state = 2
                        else:
                            state = box["state"]
                        # Traffic lights are not correctly labeled in CARLA, since they only offer the
                        # position of the base. We need some modifications and offsets.
                        x = box["position"][0]
                        y = box["position"][1]
                        z = 5.325 # in meters
                        yaw = float(box["yaw"]) + np.pi
                        x += 6 * np.cos(yaw)
                        y += 6 * np.sin(yaw)
                        items.append(["traffic_light", 
                                    0.4,   
                                    0.9, 
                                    0.4, 
                                    x,             
                                    -y, 
                                    z, 
                                    -float(box["yaw"]), 
                                    state])
                elif box["class"] == "stop_sign_vqa":
                        if not box["affects_ego"]:
                            continue
                        items.append(["stop_sign", 
                                    0.25,   
                                    0.9, 
                                    0.9, 
                                    box["position"][0],             
                                    -box["position"][1], 
                                    1.5, 
                                    -float(box["yaw"]), 
                                    box["num_points"]])
                elif box["class"] == "static_trafficwarning":
                        items.append(["static_trafficwarning", 
                                    # np.array(box["extent"][1])*2, # w 
                                    # np.array(box["extent"][2])*2, # h
                                    # np.array(box["extent"][0])*2, # l
                                    3.0,
                                    np.array(box["extent"][2])*2, # h
                                    2.5,
                                    box["position"][0],             
                                    -box["position"][1], 
                                    box["position"][2], 
                                    -float(box["yaw"]), 
                                    box["num_points"]])
                else:
                    pass

            # delete directory if exists
            # if os.path.exists(os.path.join(DATAROOT, "labels")):
            #     # delete the directory
            #     os.rmdir(os.path.join(DATAROOT, "labels"))

            # create the directory if it does not exist
            os.makedirs(os.path.join(DATAROOT, "labels"), exist_ok=True)

            with open(os.path.join(DATAROOT, "labels", ITEM + ".txt"), "w") as f:
                for item in items:
                    f.write(" ".join(map(str, item)) + "\n")

            # print("Labels created")
            # open pointcloud
            LIDAR_PATH = os.path.join(DATAROOT, "lidar", ITEM + ".laz")

            with laspy.open(LIDAR_PATH) as f:
                points = f.read()

            x = points['X']/1000 + f.header.offset[0]
            # y = -(points['Y']/1000 + f.header.offset[1])
            y = points['Y']/1000 + f.header.offset[1]
            z = points['Z']/1000 + f.header.offset[2]
            i = points['intensity']

            # save in points folder as .bin
            os.makedirs(os.path.join(DATAROOT, "points"), exist_ok=True)

            with open(os.path.join(DATAROOT, "points", ITEM + ".bin"), "wb") as f:
                np.array([x, y, z, i]).T.astype(np.float32).tofile(f)

            # adapt radar pointclouds to create only one
            RADAR_PATHS = [
                os.path.join(DATAROOT, "RADAR_FRONT", ITEM + ".bin"),
                os.path.join(DATAROOT, "RADAR_FRONT_LEFT", ITEM + ".bin"),
                os.path.join(DATAROOT, "RADAR_FRONT_RIGHT", ITEM + ".bin"),
                os.path.join(DATAROOT, "RADAR_BACK_LEFT", ITEM + ".bin"),
                os.path.join(DATAROOT, "RADAR_BACK_RIGHT", ITEM + ".bin"),
            ]

            RADARS = [
                "RADAR_FRONT",
                "RADAR_FRONT_LEFT",
                "RADAR_FRONT_RIGHT",
                "RADAR_BACK_LEFT",
                "RADAR_BACK_RIGHT",
            ]

            RADAR_MATRICES = [
                lidar2radar_front,
                lidar2radar_front_left,
                lidar2radar_front_right,
                lidar2radar_back_left,
                lidar2radar_back_right,
            ]

            ego_speed = measurements["speed"]
            full_radar_pcd = []

            for i, radar in enumerate(RADARS):
                radar_pcd = np.fromfile(os.path.join(DATAROOT, radar, ITEM + ".bin"), dtype=np.float32).reshape(-1, 4)
                ranges = radar_pcd[:, 0]
                altitude = radar_pcd[:, 1]
                azimuth = radar_pcd[:, 2]
                velocity = radar_pcd[:, 3]
                # convert to x, y, z
                x = ranges * np.cos(azimuth) * np.cos(altitude)
                y = -ranges * np.sin(azimuth) * np.cos(altitude) # VERY IMPORTANT: -y
                z = ranges * np.sin(altitude)

                # Compensate velocity
                range_unit_vector = np.array([x / ranges, y / ranges]).T

                if i == 0 or i == 3 or i == 4:
                    vr_ego = ego_speed * range_unit_vector[:, 0]
                else:
                    vr_ego = ego_speed * range_unit_vector[:, 1]

                if i == 0 or i == 2:
                    velo_comp = -velocity - vr_ego
                else:
                    velo_comp = velocity - vr_ego

                # Rotate velocity to LiDAR
                velo_comp_x = velo_comp * range_unit_vector[:, 0]
                velo_comp_y = velo_comp * range_unit_vector[:, 1]
                velo_comp_z = np.zeros_like(velo_comp)
                velo_comp_matrix = np.vstack((velo_comp_x, velo_comp_y, velo_comp_z)).T

                # Radar Rotation
                radar_rotation = RADAR_MATRICES[i][:3, :3]
                velo_comp_rotated = velo_comp_matrix @ radar_rotation.T

                # Convert to LiDAR coordinates
                radar_pcd_homogeneous = np.vstack((x, y, z, np.ones_like(x))).T

                RADAR2LIDAR = np.linalg.inv(RADAR_MATRICES[i])
                pcd_radar_lidar = radar_pcd_homogeneous @ RADAR2LIDAR.T

                # Create the final point cloud
                # add also azimuth, elevation and range
                full_radar_pcd.append(np.hstack((pcd_radar_lidar[:, :3], velo_comp_rotated[:, :2], velocity[:, np.newaxis], azimuth[:, np.newaxis], altitude[:, np.newaxis], ranges[:, np.newaxis])))
                
            # Concatenate all radar point clouds
            full_radar_pcd = np.vstack(full_radar_pcd)

            # Save the radar point cloud
            os.makedirs(os.path.join(DATAROOT, "radar_points"), exist_ok=True)
            with open(os.path.join(DATAROOT, "radar_points", ITEM + ".bin"), "wb") as f:
                full_radar_pcd.astype(np.float32).tofile(f)

    print(" ---- CREATING THE SPLITS ---- ")

    # Creating the splits
    total_files = 0

    # Check if train and val files exist
    # if not, create them empty
    if not os.path.exists(os.path.join(DATASET_PATH, "train.txt")):
        with open(os.path.join(DATASET_PATH, "train.txt"), "w") as f:
            pass
    else:
        os.remove(os.path.join(DATASET_PATH, "train.txt"))
    if not os.path.exists(os.path.join(DATASET_PATH, "val.txt")):
        with open(os.path.join(DATASET_PATH, "val.txt"), "w") as f:
            pass
    else:
        os.remove(os.path.join(DATASET_PATH, "val.txt"))
    
    if len(route_names) == 1:
        # firstly, get the number of the items in the train random split
        DATAROOT = route
        BOXES_PATH = os.path.join(DATAROOT, "boxes")
        MEASUREMENTS_PATH = os.path.join(DATAROOT, "measurements")

        # CHECK COLLISIONS

        # read number of files in the folders
        n_files = len([name for name in os.listdir(BOXES_PATH) if os.path.isfile(os.path.join(BOXES_PATH, name))])

        total_files += n_files

        # get the random split
        train_files = random.sample(range(n_files), int(n_files*0.75))
        val_files = [i for i in range(n_files) if i not in train_files]

        # order files
        train_files = sorted(train_files)
        val_files = sorted(val_files)

        # from dataroot, get only the folder name
        folder_name = os.path.basename(DATAROOT)

        # FILTER OUT COLLISIONS
        invalid_files = check_collisions(DATAROOT)
        
        # remove invalid files from train and val
        train_files = [i for i in train_files if str(i).zfill(4) not in invalid_files]
        val_files = [i for i in val_files if str(i).zfill(4) not in invalid_files]

        with open(os.path.join(DATASET_PATH, "train.txt"), "a") as f:
            for item in train_files:
                f.write(os.path.join(DATAROOT, str(item).zfill(4)) + "\n")

        with open(os.path.join(DATASET_PATH, "val.txt"), "a") as f:
            for item in val_files:
                f.write(os.path.join(DATAROOT, str(item).zfill(4)) + "\n")

    else:
        train_folders = []
        val_folders = []
        total_folders = len(route_names)

        route_names = sorted(route_names, key=lambda x: int(x.split("route_route")[1].split("_")[0]))

        train_folders = route_names[:int(total_folders*0.75)]
        val_folders = route_names[int(total_folders*0.75):]

        print("Train folders: ", len(train_folders))
        for train_folder in train_folders:
            print("Train folder: ", train_folder.split("/")[-1])
            # firstly, get the number of the items in the train random split
            DATAROOT = train_folder
            BOXES_PATH = os.path.join(DATAROOT, "boxes")
            MEASUREMENTS_PATH = os.path.join(DATAROOT, "measurements")

            # read number of files in the folders
            n_files = len([name for name in os.listdir(BOXES_PATH) if os.path.isfile(os.path.join(BOXES_PATH, name))])

            # CHECK COLLISIONS
            invalid_files = check_collisions(DATAROOT)
            # remove invalid files from train
            for i in invalid_files:
                if i in train_folder:
                    invalid_files.remove(i)
            train_files = [i for i in range(n_files) if str(i).zfill(4) not in invalid_files]

            # from dataroot, get only the folder name
            folder_name = os.path.basename(DATAROOT)

            with open(os.path.join(DATASET_PATH, "train.txt"), "a") as f:
                for item in train_files:
                    f.write(os.path.join(DATAROOT, str(item).zfill(4)) + "\n")

        print("\nVal folders: ", len(val_folders))
        for val_folder in val_folders:
            print("Val folder: ", val_folder.split("/")[-1])
            # firstly, get the number of the items in the train random split
            DATAROOT = val_folder
            BOXES_PATH = os.path.join(DATAROOT, "boxes")
            MEASUREMENTS_PATH = os.path.join(DATAROOT, "measurements")

            # read number of files in the folders
            n_files = len([name for name in os.listdir(BOXES_PATH) if os.path.isfile(os.path.join(BOXES_PATH, name))])

            # CHECK COLLISIONS
            invalid_files = check_collisions(DATAROOT)
            # remove invalid files from val
            for i in invalid_files:
                if i in val_folder:
                    invalid_files.remove(i)
            val_files = [i for i in range(n_files) if str(i).zfill(4) not in invalid_files]

            # from dataroot, get only the folder name
            folder_name = os.path.basename(DATAROOT)
            # print(folder_name)

            with open(os.path.join(DATASET_PATH, "val.txt"), "a") as f:
                for item in val_files:
                    f.write(os.path.join(DATAROOT, str(item).zfill(4)) + "\n")

    # CHECKING
    with open(os.path.join(DATASET_PATH, "train.txt"), "r") as f:
        train_files = f.readlines()
        train_files = [i.strip() for i in train_files]
        print("\nNumber of train files: ", len(train_files))
    
    with open(os.path.join(DATASET_PATH, "val.txt"), "r") as f:
        val_files = f.readlines()
        val_files = [i.strip() for i in val_files]
        print("Number of val files: ", len(val_files))

    total_files = len(train_files) + len(val_files)
    print("Total samples after filtering: ", total_files)

    
