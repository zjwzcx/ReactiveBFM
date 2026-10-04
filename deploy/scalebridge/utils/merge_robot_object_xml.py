
from pathlib import Path


def merge_robot_object_xml(orig_xml_path, metadata_dict):
    with open(orig_xml_path, 'r') as f:
        robot_xml = f.read()

    object_names = metadata_dict["object_names"] # (num_obj)
    object_scales = metadata_dict["object_scales"]  # (num_obj, 3)
    object_meshes = metadata_dict["object_meshes"] # (num_obj)
    object_props = metadata_dict["object_props"] # (num_obj)
    object_density = metadata_dict["object_density"] # (num_obj)
    object_static_pos = metadata_dict["object_static_pos"] # (num_obj, 3)
    object_static_quat = metadata_dict["object_static_quat"] # (num_obj, 4)
    num_object = len(object_names)

    for i in range(num_object):

        name = object_names[i]
        scale = object_scales[i]
        mesh = object_meshes[i]
        prop = object_props[i]
        density = object_density[i]
        static_pos = object_static_pos[i]
        static_quat = object_static_quat[i]

        asset_point = robot_xml.find("</asset>")
        insertion_content = f'  <mesh name="{name}_mesh" file="{mesh}" scale="{scale[0]} {scale[1]} {scale[2]}"/>\n '
        robot_xml = robot_xml[:asset_point]+ insertion_content + robot_xml[asset_point:]

        world_body_point = robot_xml.find("</worldbody>")

        if prop == "body":
            insertion_content = f"""  <body name="{name}" >
            <freejoint name="{name}_joint"/>
            <geom type="mesh" mesh="{name}_mesh" rgba="0.7 0.7 0.7 1" density="{density}"/>
        </body>\n  """
        elif prop == "geom":
            insertion_content = f"""<geom name="{name}" type="mesh" mesh="{name}_mesh" rgba="0.7 0.7 0.7 1" density="{density}" pos="{static_pos[0]} {static_pos[1]} {static_pos[2]}" quat="{static_quat[0]} {static_quat[1]} {static_quat[2]} {static_quat[3]}"/>\n"""
        else:
            raise NotImplementedError
        robot_xml = robot_xml[:world_body_point] + insertion_content + robot_xml[world_body_point:]

    orig_xml_path = Path(orig_xml_path)
    tmp_xml_dir = orig_xml_path.parent
    tmp_xml_name = orig_xml_path.stem
    tmp_xml_path = str(tmp_xml_dir / f"{tmp_xml_name}_{'_'.join(object_names)}.xml")
            
    with open(tmp_xml_path, "w") as f:
        f.write(robot_xml)

    return tmp_xml_path
