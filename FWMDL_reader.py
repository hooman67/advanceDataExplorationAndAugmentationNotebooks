import argparse
import glob
import os
import zlib
import cv2


def check_if_folder_exists(directory):
    try:
        os.stat(directory)
    except FileNotFoundError:
        os.makedirs(directory)


class ParseFWMDL:
    def __init__(self, folder, log_type):
        self.folder = folder
        self.log_type = log_type
        self.yaml_nodes = ["Frame", "NetOut"]
        if self.log_type == "wmdl":
            self.yaml_nodes.extend(["ToothTips"])
        else:
            self.yaml_nodes.extend(["OpticalFlowMagnitude", "OpticalFlowAngle", "Contour", "BoundingBox", "ROI", "BucketWidth"])

    @property
    def folder(self):
        return self._folder

    @folder.setter
    def folder(self, value):
        assert os.path.exists(os.path.normpath(value)), "folder not found"
        self._folder = os.path.normpath(value)

    @property
    def log_type(self):
        return self._log_type

    @log_type.setter
    def log_type(self, value):
        assert value.lower() in ["wmdl", "fmdl"], "input log type is not correct"
        self._log_type = value.lower()

    @staticmethod
    def save_yaml(yaml_name, data):
        file = open(yaml_name, "wb")
        file.write(data)
        file.close()

    def save_yaml_file_of_dl_logs(self):
        logs = glob.glob(os.path.join(self.folder, "*.{0}".format(self.log_type)))
        for log in logs:
            dl_log = open(log, 'rb').read()
            decompressed_log = zlib.decompress(dl_log).decode("ascii")

            self.save_yaml(log.replace(".{0}".format(self.log_type), ".yaml"), decompressed_log)
            print("{0} saved!".format(os.path.basename(log)))

    def save_info_of_yaml_file(self):
        yaml_files = glob.glob(os.path.join(self.folder, "*.yaml"))
        for yaml_file in yaml_files:
            print("* {0}".format(os.path.basename(yaml_file)))
            name = os.path.basename(yaml_file)
            fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_READ)
            for node in self.yaml_nodes:
                try:
                    info = fs.getNode(node)
                    if node in ["Frame", "NetOut", "OpticalFlowMagnitude", "OpticalFlowAngle"]:
                        info = info.mat()
                        if node != "Frame":
                            info = cv2.normalize(info, None, 0, 255.0, cv2.NORM_MINMAX, cv2.CV_8UC1)
                        check_if_folder_exists(os.path.join(self.folder, node))
                        cv2.imwrite(os.path.join(self.folder, node, name.replace(".yaml", ".png")), info)
                    else:
                        node_list = []
                        for i in range(0, info.size()):
                            node_list.append(info.at(i).real())
                        print("   > {0} = {1}".format(node,node_list))
                        if self.log_type == "wmdl" and node == "ToothTips":
                            number_of_tooth = int(len(node_list)/2)
                            for i in range (number_of_tooth):
                                if "Contour{0}".format(i+1) not in self.yaml_nodes:
                                    self.yaml_nodes.extend(["Contour{0}".format(i+1)])
                except TypeError:
                    continue
            fs.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, help="whether logs are fmdl or wmdl")
    parser.add_argument("--folder", type=str, help="path to logs")
    args = parser.parse_args()
    #try:
    ParseFWMDL(args.folder, args.type).save_yaml_file_of_dl_logs()
    ParseFWMDL(args.folder, args.type).save_info_of_yaml_file()
    #except Exception as message:
    print(message)
