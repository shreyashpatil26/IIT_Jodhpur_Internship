import numpy as np
import pyrealsense2 as rs
from matplotlib import pyplot as plt
import cv2
import open3d as o3d
from github_realsense_depth import DepthCamera
from github_utils import createPointCloudO3D


resolution_width, resolution_height = (640, 480)


class DepthCamera:
    def __init__(self, resolution_width, resolution_height):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        #print(img_width, img_height)
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        depth_sensor = device.first_depth_sensor()
        # Get depth scale of the device
        self.depth_scale =  depth_sensor.get_depth_scale()
            # Create an align object
        align_to = rs.stream.color

        self.align = rs.align(align_to)
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        print("device product line:", device_product_line)
        config.enable_stream(rs.stream.depth,  resolution_width,  resolution_height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color,  resolution_width,  resolution_height, rs.format.bgr8, 30)
        
        # Start streaming
        self.pipeline.start(config)
       
    def get_frame(self):
    
        # Align the depth frame to color frame
        
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        if not depth_frame or not color_frame:
            return False, None, None
        return True, depth_image, color_image

    def get_raw_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return False, None, None
        return True, depth_frame, color_frame
    
    def get_depth_scale(self):
        """
        "scaling factor" refers to the relation between depth map units and meters; 
        it has nothing to do with the focal length of the camera.
        Depth maps are typically stored in 16-bit unsigned integers at millimeter scale, thus to obtain Z value in meters, the depth map pixels need to be divided by 1000.
        """
        return self.depth_scale

    def release(self):
        self.pipeline.stop()


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
    o3d.io.write_point_cloud("o5d.ply", pcd)
        
    # At the specified voxel size down sampling
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=1)
    return pcd
    





def main():

    Realsensed435Cam = DepthCamera(resolution_width, resolution_height)

    while True:

        ret , depth_raw_frame, color_raw_frame = Realsensed435Cam.get_raw_frame()
        if not ret:
            print("Unable to get a frame")
        
        #o3D library for construct point clouds with rgbd image and camera matrix
        pcd = createPointCloudO3D(color_raw_frame, depth_raw_frame)
        
    
        color_frame = np.asanyarray(color_raw_frame.get_data())
        depth_frame = np.asanyarray(depth_raw_frame.get_data())
        print("frame shape:", color_frame.shape)
        cv2.imshow("Frame",  color_frame )
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.imwrite("frame5_color.png", color_frame)
            plt.imsave("frame5_depth.png", depth_frame)
            break
    Realsensed435Cam.release() # release rs pipeline
    o3d.visualization.draw_geometries([pcd]) 



if __name__ == '__main__':
    main()
