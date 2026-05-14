import re

with open("test_basic.py", "r") as f:
    content = f.read()

# Change the test image size to match the swin input resolution of 224x224
# or another size that swin module supports.
content = content.replace("cover = torch.rand(2, 3, 64, 64)", "cover = torch.rand(2, 3, 224, 224)")
content = content.replace("wm = torch.rand(2, 3, 64, 64)", "wm = torch.rand(2, 3, 224, 224)")

with open("test_basic.py", "w") as f:
    f.write(content)
