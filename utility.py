import datetime
import torch
from collections import deque
import os


def save_paras(model, path, filename):
    if not os.path.exists(path):
        os.mkdir(path)

    if path.endswith("/"):
        pth = path + filename
    else:
        pth = path + f"/{filename}"

    torch.save(model.state_dict(), pth)


def print_bar():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=========="*15 + f" {current_time}")


class ModelQ:
    def __init__(self, k):
        self.queue = deque()
        self.k = k

    def __repr__(self):
        return f"{self.queue}"

    def size(self):
        size = len(self.queue)
        return size

    # Main method of this class
    def stack(self, model):
        if len(self.queue) < self.k:
            self.queue.append(model)
        else:
            self.queue.popleft()
            self.queue.append(model)


# Used to delete files with specified chars
def del_files(path, chars):
    del_file = []
    files = os.listdir(path)

    for file in files:
        if file.find(chars) != -1:
            del_file.append(file)

    for f in del_file:
        del_path = os.path.join(path, f)

        # Check deleting a file
        if os.path.isfile(del_path):
            os.remove(del_path)
