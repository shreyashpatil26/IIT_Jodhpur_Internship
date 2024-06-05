import numpy as np
import pyrealsense2 as rs
from matplotlib import pyplot as plt
import cv2
import open3d as o3d


def get_intrinsic_matrix(frame, imwidth,  imheight):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    out = o3d.camera.PinholeCameraIntrinsic(imwidth,  imheight, intrinsics.fx,
                                            intrinsics.fy, intrinsics.ppx,
                                            intrinsics.ppy)
    return out



def createPointCloudO3D(color_frame, depth_frame):
    
    color_np = np.asanyarray(color_frame.get_data())
    imwidth,  imheight, channel = color_np.shape
    color_image = o3d.geometry.Image(color_np)
    
    depth_image = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))
    #if we want to get colored point clouds covert_rgb_to_intensity should be false
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image,
        convert_rgb_to_intensity=True) # Generate colored pointclouds

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, get_intrinsic_matrix(color_frame,imwidth,  imheight))
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    # Normal calculation
    pcd.estimate_normals()
    
    # Save point clouds as ply format
    o3d.io.write_point_cloud("o355555d.ply", pcd)
        
    # At the specified voxel size down sampling
    #voxel_down_pcd = pcd.voxel_down_sample(voxel_size=1)
    return pcd
    

