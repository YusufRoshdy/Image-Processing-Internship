# Augmentation: Fog generation

after trying diffrent libararies, I found the best one that works is the wither libarary from imgaug.
I scan the the input using cv2.imread()

```python
imgs = []
for i in inputs:
    image = cv2.imread(i)
    if image is not 'None':
        imgs.append(image)
```
then looping on the images to augment them by calling fog_augmentation1()
```python
i = 0
while i < len(imgs):
    cv2.imwrite(outputs[i], fog_augmentation1(imgs[i]))
    i = i+1
```
in the function fog_augmentation1() it takes a parameter img and return it after adding the fog using iaa.Sequential(iaa.Fog(random_state=9)) and after trying different values for random_stat the best one that worked is 9
```python
def fog_augmentation1(img):
    seq = iaa.Sequential(iaa.Fog(random_state=9))
    img = seq.augment_image(img)
    seq = iaa.Sequential(iaa.MotionBlur())
    img = seq.augment_image(img)
    return img
```

The results are shown below:

<img align="left" width="150" height="100" src="https://user-images.githubusercontent.com/24385400/56155785-62f59080-5fc4-11e9-921d-688d8597e098.jpg">
<img align="left" width="150" height="100" src="https://user-images.githubusercontent.com/24385400/56155786-62f59080-5fc4-11e9-990e-302dfdb47c2d.jpg">
<img align="left" width="150" height="100" src="https://user-images.githubusercontent.com/24385400/56155788-62f59080-5fc4-11e9-8e00-242bd6fdd84a.jpg">
<img align="left" width="150" height="100" src="https://user-images.githubusercontent.com/24385400/56155777-61c46380-5fc4-11e9-9ee8-ba4aa96f734a.jpg">
<img align="left" width="150" height="100" src="https://user-images.githubusercontent.com/24385400/56155778-61c46380-5fc4-11e9-93a1-0b0363c33327.jpg">


<img align="left" width="150" height="100" src="https://user-images.githubusercontent.com/24385400/56155734-448f9500-5fc4-11e9-8f30-325303b5fa7e.jpg">
<img align="left" width="150" height="100" src="https://user-images.githubusercontent.com/24385400/56155735-45282b80-5fc4-11e9-8549-fe8de51af8fe.jpg">
<img align="left" width="150" height="100" src="https://user-images.githubusercontent.com/24385400/56155727-43f6fe80-5fc4-11e9-83c8-eea2bcb160e9.jpg">
<img align="left" width="150" height="100" src="https://user-images.githubusercontent.com/24385400/56155729-43f6fe80-5fc4-11e9-883f-f911996b8108.jpg">
<img align="left" width="150" height="100" src="https://user-images.githubusercontent.com/24385400/56155730-43f6fe80-5fc4-11e9-8ac0-133c76767508.jpg"><br />


# Makeup: Facial keypoints detection & drawing
I wrote a code that add two stars and a moustache on the face, by using the pre-trained facial landmark detector inside the dlib library is used to estimate the location of 68 (x, y)-coordinates that map to facial structures on the face.

The indexes of the 68 coordinates can be visualized on the image below:
<img src="https://user-images.githubusercontent.com/24385400/56217881-a1df2100-606c-11e9-9e50-964274dc5fa2.jpg"><br />

with the above points we can calculate the angle and position of the stars and the moustache  
and by using a formula to overlay them on the face without the background

and the result are shown below:

<img align="left" width="150" height="100" src="https://user-images.githubusercontent.com/24385400/56219466-790c5b00-606f-11e9-8c23-2b04291a3883.jpg">
<img align="left" width="150" height="100" src="https://user-images.githubusercontent.com/24385400/56219467-790c5b00-606f-11e9-82c2-46a04edb280d.jpg">
<img align="left" width="150" height="100" src="https://user-images.githubusercontent.com/24385400/56219468-79a4f180-606f-11e9-8f6d-715c197db472.jpg">
<img align="left" width="150" height="100" src="https://user-images.githubusercontent.com/24385400/56219469-79a4f180-606f-11e9-96f9-f7a37248c525.jpg">
<img align="left" width="150" height="100" src="https://user-images.githubusercontent.com/24385400/56219470-79a4f180-606f-11e9-8bf8-a13a0691d785.jpg">


<img align="left" width="150" height="100" src="https://user-images.githubusercontent.com/24385400/56219471-79a4f180-606f-11e9-8700-7765c2542d82.jpg">
<img align="left" width="150" height="100" src="https://user-images.githubusercontent.com/24385400/56219472-7a3d8800-606f-11e9-9e6f-157228d3bb12.jpg">
<img align="left" width="150" height="100" src="https://user-images.githubusercontent.com/24385400/56219474-7a3d8800-606f-11e9-9ed4-c65e6bcc1d39.jpg">
<img align="left" width="150" height="100" src="https://user-images.githubusercontent.com/24385400/56219475-7a3d8800-606f-11e9-961c-b20b36cf7744.jpg">
<img align="left" width="150" height="100" src="https://user-images.githubusercontent.com/24385400/56219476-7a3d8800-606f-11e9-9de5-dd724fda8de4.jpg">
