# Hexapod133a

## Demo Videos:
[The demo videos can be found here](https://drive.google.com/drive/u/3/folders/1ilrjRt14x5afsUi1LaYct1I82j9jsDka)

## Build Instructions:

### Build:
```bash
cd ~/robotws
colcon build --symlink-install
```

### Source the Workspace:
```bash
source ~/robotws/install/setup.bash
```

### To launch the code
```bash
ros2 launch hexapod_demo full_move.launch.py
```
