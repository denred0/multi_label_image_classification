import albumentations as A
import cv2
import numpy as np

image = cv2.imread('data/albumentations/stone5.jpg', cv2.IMREAD_COLOR)

# resize to square
h, w = image.shape[:2]
x_center = w // 2
y_center = h // 2
image = image[y_center - x_center + 10:y_center + x_center + 10, :]

size = 320
image = cv2.resize(image, (size, size))

augments = []
images = []
augments.append('Original')
images.append(image)

transform = A.HorizontalFlip(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('HorizontalFlip')
images.append(augmented_image)

transform = A.VerticalFlip(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('VerticalFlip')
images.append(augmented_image)

transform = A.Transpose(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('Transpose')
images.append(augmented_image)

transform = A.RandomRotate90(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('RandomRotate90')
images.append(augmented_image)

transform = A.Flip(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('Flip')
images.append(augmented_image)

transform = A.MotionBlur(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('MotionBlur')
images.append(augmented_image)

transform = A.MedianBlur(blur_limit=7, p=1.0)
augmented_image = transform(image=image)['image']
augments.append('MedianBlur blur_limit=7')
images.append(augmented_image)

transform = A.MedianBlur(blur_limit=3, p=1.0)
augmented_image = transform(image=image)['image']
augments.append('MedianBlur blur_limit=3')
images.append(augmented_image)

transform = A.Blur(blur_limit=7, p=1.0)
augmented_image = transform(image=image)['image']
augments.append('Blur blur_limit=7')
images.append(augmented_image)

transform = A.Blur(blur_limit=2, p=1.0)
augmented_image = transform(image=image)['image']
augments.append('Blur blur_limit=2')
images.append(augmented_image)

transform = A.CLAHE(clip_limit=4.0, p=1.0)
augmented_image = transform(image=image)['image']
augments.append('CLAHE clip_limit=4.0')
images.append(augmented_image)

transform = A.CLAHE(clip_limit=10.0, p=1.0)
augmented_image = transform(image=image)['image']
augments.append('CLAHE clip_limit=10.0')
images.append(augmented_image)

transform = A.Sharpen(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('Sharpen')
images.append(augmented_image)

transform = A.Emboss(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('Emboss')
images.append(augmented_image)

transform = A.RandomBrightnessContrast(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('RandomBrightnessContrast')
images.append(augmented_image)

transform = A.ColorJitter(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('ColorJitter')
images.append(augmented_image)

transform = A.Cutout(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('Cutout')
images.append(augmented_image)

transform = A.CenterCrop(p=1.0, height=256, width=256)
augmented_image = transform(image=image)['image']
transform = A.Resize(height=size, width=size, p=1.0)
augmented_image = transform(image=image)['image']
augments.append('CenterCrop')
images.append(augmented_image)

transform = A.ChannelDropout(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('ChannelDropout')
images.append(augmented_image)

transform = A.CoarseDropout(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('CoarseDropout')
images.append(augmented_image)

transform = A.ChannelShuffle(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('ChannelShuffle')
images.append(augmented_image)

transform = A.CropAndPad(px=32, p=1.0)
augmented_image = transform(image=image)['image']
augments.append('CropAndPad px=32')
images.append(augmented_image)

transform = A.Downscale(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('Downscale')
images.append(augmented_image)

transform = A.Equalize(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('Equalize')
images.append(augmented_image)

transform = A.FancyPCA(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('FancyPCA')
images.append(augmented_image)

transform = A.FromFloat(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('FromFloat')
images.append(augmented_image)

transform = A.GaussNoise(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('GaussNoise')
images.append(augmented_image)

transform = A.GridDistortion(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('GridDistortion')
images.append(augmented_image)

transform = A.HueSaturationValue(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('HueSaturationValue')
images.append(augmented_image)

transform = A.ImageCompression(quality_lower=85, p=1.0)
augmented_image = transform(image=image)['image']
augments.append('ImageCompression quality_lower=85')
images.append(augmented_image)

transform = A.InvertImg(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('InvertImg')
images.append(augmented_image)

transform = A.ISONoise(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('ISONoise')
images.append(augmented_image)

transform = A.RandomGamma(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('RandomGamma')
images.append(augmented_image)

transform = A.OpticalDistortion(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('OpticalDistortion')
images.append(augmented_image)

transform = A.RandomGridShuffle(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('RandomGridShuffle')
images.append(augmented_image)

transform = A.RGBShift(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('RGBShift')
images.append(augmented_image)

transform = A.RandomContrast(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('RandomContrast')
images.append(augmented_image)

transform = A.ToGray(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('ToGray')
images.append(augmented_image)

transform = A.ToSepia(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('ToSepia')
images.append(augmented_image)

transform = A.JpegCompression(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('JpegCompression')
images.append(augmented_image)

transform = A.ToFloat(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('ToFloat')
images.append(augmented_image)

transform = A.RandomSnow(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('RandomSnow')
images.append(augmented_image)

transform = A.RandomRain(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('RandomRain')
images.append(augmented_image)

transform = A.RandomFog(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('RandomFog')
images.append(augmented_image)

transform = A.RandomSunFlare(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('RandomSunFlare')
images.append(augmented_image)

transform = A.RandomShadow(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('RandomShadow')
images.append(augmented_image)

transform = A.RandomToneCurve(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('RandomToneCurve')
images.append(augmented_image)

transform = A.Lambda(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('Lambda')
images.append(augmented_image)

transform = A.Solarize(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('Solarize')
images.append(augmented_image)

transform = A.Posterize(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('Posterize')
images.append(augmented_image)

transform = A.MultiplicativeNoise(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('MultiplicativeNoise')
images.append(augmented_image)

transform = A.Superpixels(p=1.0)
augmented_image = transform(image=image)['image']
augments.append('Superpixels')
images.append(augmented_image)

image_per_row = 3
total_image = np.ones(
    (size * (len(images) // image_per_row) + size * min(len(images) % image_per_row, 1), size * image_per_row, 3)) * 255

for i, (aug_name, image) in enumerate(zip(augments, images)):
    image_copy = image.copy()

    cv2.putText(image_copy, str(aug_name), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    total_image[i // image_per_row * size:i // image_per_row * size + size,
    i % image_per_row * size: i % image_per_row * size + size, :] = image_copy

    # print(i)

cv2.imwrite('data/albumentations/image_aug_examples.jpg', total_image)
