import time
import sys
import math
import json

from functools import lru_cache
from typing import Any

openvr: Any = None
try:
    import openvr  # pyright: ignore[reportMissingImports]
except ImportError:
    openvr = None


# TARGET_AXIS_TRANSFORM = [
#     [0.0, 0.0, 1.0],
#     [1.0, 0.0, 0.0],
#     [0.0, 1.0, 0.0],
# ]
TARGET_AXIS_TRANSFORM = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]

def ensure_openvr_available():
    if openvr is None:
        raise ImportError("openvr is required to use vive tracking features. Install it with `pip install openvr`.")


def matrix_transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def matrix_multiply(left, right):
    return [
        [sum(left[row][index] * right[index][column] for index in range(len(right))) for column in range(len(right[0]))]
        for row in range(len(left))
    ]


def extract_position(pose_mat):
    return [pose_mat[0][3], pose_mat[1][3], pose_mat[2][3]]


def reorder_vector_axes(vector):
    return [vector[2], vector[0], vector[1]]


def transform_rotation_matrix(rotation_matrix):
    axis_transform_transpose = matrix_transpose(TARGET_AXIS_TRANSFORM)
    return matrix_multiply(matrix_multiply(TARGET_AXIS_TRANSFORM, rotation_matrix), axis_transform_transpose)


def transform_pose_matrix(pose_mat):
    rotation_matrix = [[pose_mat[row][column] for column in range(3)] for row in range(3)]
    transformed_rotation = transform_rotation_matrix(rotation_matrix)
    transformed_position = reorder_vector_axes(extract_position(pose_mat))
    return [
        transformed_rotation[0] + [transformed_position[0]],
        transformed_rotation[1] + [transformed_position[1]],
        transformed_rotation[2] + [transformed_position[2]],
    ]


def offset_position_z(position, z_offset):
    adjusted_position = list(position)
    adjusted_position[2] -= z_offset
    return adjusted_position


def vector3_to_dict(vector):
    return {"x": vector[0], "y": vector[1], "z": vector[2]}


def quaternion_to_dict(quaternion_values):
    return {
        "w": quaternion_values[3],
        "x": quaternion_values[4],
        "y": quaternion_values[5],
        "z": quaternion_values[6],
    }


def apply_relative_z_to_device_data(device_data, reference_device_name):
    if reference_device_name not in device_data:
        raise KeyError("Reference device '{}' not found in device data.".format(reference_device_name))

    reference_z = device_data[reference_device_name]["position"]["z"]
    adjusted_data = {}
    for device_name, values in device_data.items():
        adjusted_values = dict(values)
        adjusted_position = dict(values["position"])
        adjusted_position["z"] -= reference_z
        adjusted_values["position"] = adjusted_position
        adjusted_values["reference_z"] = reference_z
        adjusted_data[device_name] = adjusted_values

    return adjusted_data

# Function to print out text but instead of starting a new line it will overwrite the existing line
def update_text(txt):
    sys.stdout.write('\r'+txt)
    sys.stdout.flush()

#Convert the standard 3x4 position/rotation matrix to a x,y,z location and the appropriate Euler angles (in degrees)
def convert_to_euler(pose_mat):
    yaw = 180 / math.pi * math.atan2(pose_mat[1][0], pose_mat[0][0])
    pitch = 180 / math.pi * math.atan2(pose_mat[2][0], pose_mat[0][0])
    roll = 180 / math.pi * math.atan2(pose_mat[2][1], pose_mat[2][2])
    x = pose_mat[0][3]
    y = pose_mat[1][3]
    z = pose_mat[2][3]
    return [x,y,z, yaw, pitch, roll]

#Convert the standard 3x4 position/rotation matrix to a x,y,z location and the appropriate Quaternion
def convert_to_quaternion(pose_mat):
    # Per issue #2, adding a abs() so that sqrt only results in real numbers
    r_w = math.sqrt(abs(1 + pose_mat[0][0] + pose_mat[1][1] + pose_mat[2][2])) / 2
    r_x = (pose_mat[2][1] - pose_mat[1][2]) / (4 * r_w)
    r_y = (pose_mat[0][2] - pose_mat[2][0]) / (4 * r_w)
    r_z = (pose_mat[1][0] - pose_mat[0][1]) / (4 * r_w)

    x = pose_mat[0][3]
    y = pose_mat[1][3]
    z = pose_mat[2][3]
    return [x, y, z, r_w, r_x, r_y, r_z]


def convert_to_euler_in_target_frame(pose_mat):
    return convert_to_euler(transform_pose_matrix(pose_mat))


def convert_to_quaternion_in_target_frame(pose_mat):
    return convert_to_quaternion(transform_pose_matrix(pose_mat))

#Define a class to make it easy to append pose matricies and convert to both Euler and Quaternion for plotting
class pose_sample_buffer():
    def __init__(self):
        self.i = 0
        self.index = []
        self.time = []
        self.x = []
        self.y = []
        self.z = []
        self.yaw = []
        self.pitch = []
        self.roll = []
        self.r_w = []
        self.r_x = []
        self.r_y = []
        self.r_z = []

    def append(self,pose_mat,t):
        self.time.append(t)
        self.x.append(pose_mat[0][3])
        self.y.append(pose_mat[1][3])
        self.z.append(pose_mat[2][3])
        self.yaw.append(180 / math.pi * math.atan(pose_mat[1][0] /pose_mat[0][0]))
        self.pitch.append(180 / math.pi * math.atan(-1 * pose_mat[2][0] / math.sqrt(pow(pose_mat[2][1], 2) + math.pow(pose_mat[2][2], 2))))
        self.roll.append(180 / math.pi * math.atan(pose_mat[2][1] /pose_mat[2][2]))
        r_w = math.sqrt(abs(1+pose_mat[0][0]+pose_mat[1][1]+pose_mat[2][2]))/2
        self.r_w.append(r_w)
        self.r_x.append((pose_mat[2][1]-pose_mat[1][2])/(4*r_w))
        self.r_y.append((pose_mat[0][2]-pose_mat[2][0])/(4*r_w))
        self.r_z.append((pose_mat[1][0]-pose_mat[0][1])/(4*r_w))

def get_pose(vr_obj):
    return vr_obj.getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)


def get_device_string(vr_obj, device_index, property_id):
    value = vr_obj.getStringTrackedDeviceProperty(device_index, property_id)
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return value


class vr_tracked_device():
    def __init__(self,vr_obj,index,device_class):
        self.device_class = device_class
        self.index = index
        self.vr = vr_obj
        self.name = None

    def _get_valid_tracked_pose(self, pose=None):
        if pose is None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return pose[self.index]
        return None

    @lru_cache(maxsize=None)
    def get_serial(self):
        return get_device_string(self.vr, self.index, openvr.Prop_SerialNumber_String)

    def get_model(self):
        return get_device_string(self.vr, self.index, openvr.Prop_ModelNumber_String)

    def get_battery_percent(self):
        return self.vr.getFloatTrackedDeviceProperty(self.index, openvr.Prop_DeviceBatteryPercentage_Float)[0]

    def is_charging(self):
        return self.vr.getBoolTrackedDeviceProperty(self.index, openvr.Prop_DeviceIsCharging_Bool)[0]


    def sample(self,num_samples,sample_rate):
        interval = 1/sample_rate
        rtn = pose_sample_buffer()
        sample_start = time.time()
        for i in range(num_samples):
            start = time.time()
            pose = get_pose(self.vr)
            rtn.append(pose[self.index].mDeviceToAbsoluteTracking,time.time()-sample_start)
            sleep_time = interval- (time.time()-start)
            if sleep_time>0:
                time.sleep(sleep_time)
        return rtn

    def get_pose_euler(self, pose=None):
        tracked_pose = self._get_valid_tracked_pose(pose)
        if tracked_pose is None:
            return None
        return convert_to_euler(tracked_pose.mDeviceToAbsoluteTracking)
        
    def get_pose_matrix(self, pose=None):
        tracked_pose = self._get_valid_tracked_pose(pose)
        if tracked_pose is None:
            return None
        return tracked_pose.mDeviceToAbsoluteTracking

    def get_velocity(self, pose=None):
        tracked_pose = self._get_valid_tracked_pose(pose)
        if tracked_pose is None:
            return None
        return tracked_pose.vVelocity

    def get_angular_velocity(self, pose=None):
        tracked_pose = self._get_valid_tracked_pose(pose)
        if tracked_pose is None:
            return None
        return tracked_pose.vAngularVelocity

    def get_pose_quaternion(self, pose=None):
        tracked_pose = self._get_valid_tracked_pose(pose)
        if tracked_pose is None:
            return None
        return convert_to_quaternion(tracked_pose.mDeviceToAbsoluteTracking)

    def get_transformed_pose_matrix(self, pose=None):
        tracked_pose = self._get_valid_tracked_pose(pose)
        if tracked_pose is None:
            return None
        return transform_pose_matrix(tracked_pose.mDeviceToAbsoluteTracking)

    def get_transformed_pose_euler(self, pose=None):
        tracked_pose = self._get_valid_tracked_pose(pose)
        if tracked_pose is None:
            return None
        return convert_to_euler_in_target_frame(tracked_pose.mDeviceToAbsoluteTracking)

    def get_transformed_pose_quaternion(self, pose=None):
        tracked_pose = self._get_valid_tracked_pose(pose)
        if tracked_pose is None:
            return None
        return convert_to_quaternion_in_target_frame(tracked_pose.mDeviceToAbsoluteTracking)

    def get_transformed_velocity(self, pose=None):
        velocity = self.get_velocity(pose)
        if velocity is None:
            return None
        return reorder_vector_axes(velocity)

    def get_transformed_angular_velocity(self, pose=None):
        angular_velocity = self.get_angular_velocity(pose)
        if angular_velocity is None:
            return None
        return reorder_vector_axes(angular_velocity)

    def get_transformed_data(self, pose=None):
        transformed_pose_quaternion = self.get_transformed_pose_quaternion(pose)
        if transformed_pose_quaternion is None:
            return None

        position = transformed_pose_quaternion[:3]
        velocity = self.get_transformed_velocity(pose)
        angular_velocity = self.get_transformed_angular_velocity(pose)

        return {
            "name": self.name,
            "device_class": self.device_class,
            "index": self.index,
            "serial": self.get_serial(),
            "model": self.get_model(),
            "position": vector3_to_dict(position),
            "quaternion": quaternion_to_dict(transformed_pose_quaternion),
            "velocity": vector3_to_dict(velocity) if velocity is not None else None,
            "angular_velocity": vector3_to_dict(angular_velocity) if angular_velocity is not None else None,
        }

    def get_data(self, pose=None):
        pose_quaternion = self.get_pose_quaternion(pose)
        if pose_quaternion is None:
            return None

        position = pose_quaternion[:3]
        velocity = self.get_velocity(pose)
        angular_velocity = self.get_angular_velocity(pose)

        return {
            "name": self.name,
            "device_class": self.device_class,
            "index": self.index,
            "serial": self.get_serial(),
            "model": self.get_model(),
            "position": vector3_to_dict(position),
            "quaternion": quaternion_to_dict(pose_quaternion),
            "velocity": vector3_to_dict(velocity) if velocity is not None else None,
            "angular_velocity": vector3_to_dict(angular_velocity) if angular_velocity is not None else None,
        }

    def controller_state_to_dict(self, pControllerState):
        # This function is graciously borrowed from https://gist.github.com/awesomebytes/75daab3adb62b331f21ecf3a03b3ab46
        # docs: https://github.com/ValveSoftware/openvr/wiki/IVRSystem::GetControllerState
        d = {}
        d['unPacketNum'] = pControllerState.unPacketNum
        # on trigger .y is always 0.0 says the docs
        d['trigger'] = pControllerState.rAxis[1].x
        # 0.0 on trigger is fully released
        # -1.0 to 1.0 on joystick and trackpads
        d['trackpad_x'] = pControllerState.rAxis[0].x
        d['trackpad_y'] = pControllerState.rAxis[0].y
        # These are published and always 0.0
        # for i in range(2, 5):
        #     d['unknowns_' + str(i) + '_x'] = pControllerState.rAxis[i].x
        #     d['unknowns_' + str(i) + '_y'] = pControllerState.rAxis[i].y
        d['ulButtonPressed'] = pControllerState.ulButtonPressed
        d['ulButtonTouched'] = pControllerState.ulButtonTouched
        # To make easier to understand what is going on
        # Second bit marks menu button
        d['menu_button'] = bool(pControllerState.ulButtonPressed >> 1 & 1)
        # 32 bit marks trackpad
        d['trackpad_pressed'] = bool(pControllerState.ulButtonPressed >> 32 & 1)
        d['trackpad_touched'] = bool(pControllerState.ulButtonTouched >> 32 & 1)
        # third bit marks grip button
        d['grip_button'] = bool(pControllerState.ulButtonPressed >> 2 & 1)
        # System button can't be read, if you press it
        # the controllers stop reporting
        return d

    def get_controller_inputs(self):
        result, state = self.vr.getControllerState(self.index)
        return self.controller_state_to_dict(state)

class vr_tracking_reference(vr_tracked_device):
    def get_mode(self):
        return get_device_string(self.vr, self.index, openvr.Prop_ModeLabel_String).upper()
    def sample(self,num_samples,sample_rate):
        print("Warning: Tracking References do not move, sample isn't much use...")

class triad_openvr():
    def __init__(self, configfile_path=None):
        # Initialize OpenVR in the
        ensure_openvr_available()
        self.vr = openvr.init(openvr.VRApplication_Other)
        self.vrsystem = openvr.VRSystem()

        # Initializing object to hold indexes for various tracked objects
        self.object_names = {"Tracking Reference":[],"HMD":[],"Controller":[],"Tracker":[]}
        self.devices = {}
        self.device_index_map = {}
        poses = self.vr.getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0,
                                                               openvr.k_unMaxTrackedDeviceCount)

        # Loading config file
        if configfile_path:
            try:
                with open(configfile_path, 'r') as json_data:
                    config = json.load(json_data)
            except EnvironmentError: # parent of IOError, OSError *and* WindowsError where available
                print('config.json not found.')
                exit(1)

            # Iterate through the pose list to find the active devices and determine their type
            for i in range(openvr.k_unMaxTrackedDeviceCount):
                if poses[i].bDeviceIsConnected:
                    device_serial = get_device_string(self.vr, i, openvr.Prop_SerialNumber_String)
                    for device in config['devices']:
                        if device_serial == device['serial']:
                            device_name = device['name']
                            self.object_names[device['type']].append(device_name)
                            self.devices[device_name] = vr_tracked_device(self.vr,i,device['type'])
                            self.devices[device_name].name = device_name
        else:
            # Iterate through the pose list to find the active devices and determine their type
            for i in range(openvr.k_unMaxTrackedDeviceCount):
                if poses[i].bDeviceIsConnected:
                    self.add_tracked_device(i)

    def __del__(self):
        openvr.shutdown()

    def get_pose(self):
        return get_pose(self.vr)

    def poll_vr_events(self):
        """
        Used to poll VR events and find any new tracked devices or ones that are no longer tracked.
        """
        event = openvr.VREvent_t()
        while self.vrsystem.pollNextEvent(event):
            if event.eventType == openvr.VREvent_TrackedDeviceActivated:
                self.add_tracked_device(event.trackedDeviceIndex)
            elif event.eventType == openvr.VREvent_TrackedDeviceDeactivated:
                #If we were already tracking this device, quit tracking it.
                if event.trackedDeviceIndex in self.device_index_map:
                    self.remove_tracked_device(event.trackedDeviceIndex)

    def add_tracked_device(self, tracked_device_index):
        i = tracked_device_index
        device_class = self.vr.getTrackedDeviceClass(i)
        if (device_class == openvr.TrackedDeviceClass_Controller):
            device_name = "controller_"+str(len(self.object_names["Controller"])+1)
            self.object_names["Controller"].append(device_name)
            self.devices[device_name] = vr_tracked_device(self.vr,i,"Controller")
            self.devices[device_name].name = device_name
            self.device_index_map[i] = device_name
        elif (device_class == openvr.TrackedDeviceClass_HMD):
            device_name = "hmd_"+str(len(self.object_names["HMD"])+1)
            self.object_names["HMD"].append(device_name)
            self.devices[device_name] = vr_tracked_device(self.vr,i,"HMD")
            self.devices[device_name].name = device_name
            self.device_index_map[i] = device_name
        elif (device_class == openvr.TrackedDeviceClass_GenericTracker):
            device_name = "tracker_"+str(len(self.object_names["Tracker"])+1)
            self.object_names["Tracker"].append(device_name)
            self.devices[device_name] = vr_tracked_device(self.vr,i,"Tracker")
            self.devices[device_name].name = device_name
            self.device_index_map[i] = device_name
        elif (device_class == openvr.TrackedDeviceClass_TrackingReference):
            device_name = "tracking_reference_"+str(len(self.object_names["Tracking Reference"])+1)
            self.object_names["Tracking Reference"].append(device_name)
            self.devices[device_name] = vr_tracking_reference(self.vr,i,"Tracking Reference")
            self.devices[device_name].name = device_name
            self.device_index_map[i] = device_name

    def remove_tracked_device(self, tracked_device_index):
        if tracked_device_index in self.device_index_map:
            device_name = self.device_index_map[tracked_device_index]
            self.object_names[self.devices[device_name].device_class].remove(device_name)
            del self.device_index_map[tracked_device_index]
            del self.devices[device_name]
        else:
            raise Exception("Tracked device index {} not valid. Not removing.".format(tracked_device_index))

    def rename_device(self,old_device_name,new_device_name):
        self.devices[new_device_name] = self.devices.pop(old_device_name)
        self.devices[new_device_name].name = new_device_name
        for i in range(len(self.object_names[self.devices[new_device_name].device_class])):
            if self.object_names[self.devices[new_device_name].device_class][i] == old_device_name:
                self.object_names[self.devices[new_device_name].device_class][i] = new_device_name

    def get_device_names(self):
        return list(self.devices.keys())

    def get_devices_data(self, device_names=None):
        if device_names is None:
            device_names = self.get_device_names()

        missing_devices = [device_name for device_name in device_names if device_name not in self.devices]
        if missing_devices:
            raise KeyError("Unknown device(s): {}".format(", ".join(missing_devices)))

        pose = self.get_pose()
        device_data = {}
        for device_name in device_names:
            raw_data = self.devices[device_name].get_data(pose=pose)
            if raw_data is not None:
                device_data[device_name] = raw_data

        return device_data

    def get_transformed_devices_data(self, device_names=None, relative_to_first_device_z=False):
        if device_names is None:
            device_names = self.get_device_names()

        missing_devices = [device_name for device_name in device_names if device_name not in self.devices]
        if missing_devices:
            raise KeyError("Unknown device(s): {}".format(", ".join(missing_devices)))

        pose = self.get_pose()
        device_data = {}
        for device_name in device_names:
            transformed_data = self.devices[device_name].get_transformed_data(pose=pose)
            if transformed_data is not None:
                device_data[device_name] = transformed_data

        if relative_to_first_device_z and device_data:
            first_available_name = next(iter(device_data))
            device_data = apply_relative_z_to_device_data(device_data, first_available_name)

        return device_data

    def print_discovered_objects(self):
        for device_type in self.object_names:
            plural = device_type
            if len(self.object_names[device_type])!=1:
                plural+="s"
            print("Found "+str(len(self.object_names[device_type]))+" "+plural)
            for device in self.object_names[device_type]:
                if device_type == "Tracking Reference":
                    print("  "+device+" ("+self.devices[device].get_serial()+
                          ", Mode "+self.devices[device].get_model()+
                          ", "+self.devices[device].get_model()+
                          ")")
                else:
                    print("  "+device+" ("+self.devices[device].get_serial()+
                          ", "+self.devices[device].get_model()+")")
