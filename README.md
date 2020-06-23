# Pyramids-and-Optic-Flow--CV
## implementation the Lucas Kanade algorithm.
![optical ﬂow](https://user-images.githubusercontent.com/44798397/85467866-96389580-b560-11ea-90e4-defa5d03f5cc.jpg)



### Given two images, returns the Translation from im1 to im2
### :param im1: Image 1
### :param im2: Image 2
### :param step_size: The image sample size:
### :param win_size: The optical flow window size (odd number)
### :return: Original points [[y,x]...], [[dU,dV]...] for each points


![WhatsApp Image 2020-06-21 at 11 22 01](https://user-images.githubusercontent.com/44798397/85468083-d8fa6d80-b560-11ea-9613-afedf31c60b8.jpeg)


## Gaussian pyramid

### """
### Creates a Gaussian Pyramid
### :param img: Original image
### :param levels: Pyramid depth
### :return: Gaussian pyramid (list of images)
### """

![pyramid](https://user-images.githubusercontent.com/44798397/85468857-d51b1b00-b561-11ea-9a3e-821fe66be562.jpg)

## Laplacian Pyramids

### """
### Creates a Laplacian pyramid
### :param img: Original image
### :param levels: Pyramid depth
### :return: Laplacian Pyramid (list of images)
### """

![Laplacian](https://user-images.githubusercontent.com/44798397/85469103-1dd2d400-b562-11ea-8e70-b37e12587c43.jpg)

## Pyramid Blending
"""
### Blends two images using PyramidBlend method
### """"
### :param img_1: Image 1
### :param img_2: Image 2
### :param mask: Blend mask
### :param levels: Pyramid depth
### :return: (Naive blend, Blended Image)
### """
### The Naive blend, is blending without using the pyramid.

![blending](https://user-images.githubusercontent.com/44798397/85469309-5a9ecb00-b562-11ea-97ac-da2fd470a24d.jpg)




