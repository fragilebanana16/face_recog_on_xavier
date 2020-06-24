# 1.Introduction
some works on Xavier, contains:
- daface recog
- dlib face recog
- opencv contrib saliency
# 2.Requirements
- opencv contrib saliency
- dface sdk and its activation code
- dlib
# 3.Usage
cmakelists and src are provided whereas envirments and models and other libraries are not. Refer to the official website of dlib, opencv, dface to obtain them. 
```
cd build
cmake -build ..
make
./ELF
```

# 4.Results
#### 4.1 dface recog Charlie Puth
![image](https://github.com/fragilebanana16/face_recog_on_xavier/blob/master/screenshots/dface_find_chalie_puth1.png)

![image](https://github.com/fragilebanana16/face_recog_on_xavier/blob/master/screenshots/dface_find_charlie_puth2.png)

![image](https://github.com/fragilebanana16/face_recog_on_xavier/blob/master/screenshots/dface_find_charlie_puth3.png)

#### 4.2 dlib face detection
![image](https://github.com/fragilebanana16/face_recog_on_xavier/blob/master/screenshots/dlib_face.png)

#### 4.3 cv orb pts on building
![image](https://github.com/fragilebanana16/face_recog_on_xavier/blob/master/screenshots/sal_building_most_pts.png)

#### 4.4 rasterize
![image](https://github.com/fragilebanana16/face_recog_on_xavier/blob/master/screenshots/rasterize_case.png)

#### 4.5 pytorch saliency(slow 5 fps on Mac book)
![image](https://github.com/fragilebanana16/face_recog_on_xavier/blob/master/screenshots/pytorch_sal_slow_dog.png)

#### 4.6 yolo3 on face detection(12 fps)
![image](https://github.com/fragilebanana16/face_recog_on_xavier/blob/master/screenshots/yolo3_face.png)

