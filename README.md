# Diff

## Data structure
```
- trans_data
|- (dataset)_(scene)_(num)_poses.pt
|- (dataset)_(scene)_(num)_poses.pt
|- (dataset)_(scene)_(num)_poses.pt
|- (dataset)_(scene)_(num)_poses.pt
|- (dataset)_(scene)_(num)_poses.pt
```

```
'kp3d'              : [T, 17, 3]
'kp3d_vel'          : [T, 17, 3]
'root_pose'         : [T, 3]
'body_pose'         : [T, 69]
'root_rotmat'       : [T, 3, 3]
'body_rotmat'       : [T, 23, 3, 3]
'root_rotmat_vel'   : [T, 3]
'trans'             : [T, 3]
'trans_vel'         : [T, 3]
'shape'             : [T, 10]
'contacts'          : [T, 2]
'world2aligned_rot' : [T, 3, 3]
```
kp3d : COCO type, World coordinate