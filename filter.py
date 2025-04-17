import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


def color_compress(pic, k):
    img = Image.open(pic)
    matrix = np.array(img)   #getting the matrix of the image
    
    color_dict = {
        "red":   matrix[:, :, 0],  #the red component of each pixel of the image
        "green": matrix[:, :, 1],   #green component
        "blue":  matrix[:, :, 2]     #blue
    }
    for color, channel in color_dict.items():
        color_dict[color] = svd_compress(channel, k)   #svd compressing each color's matrix

    compressed_image = np.stack([color_dict["red"], color_dict["green"], color_dict["blue"]], axis=2)  #re-combining the colors into 1 matrix
    compressed_image = np.clip(compressed_image, 0, 255).astype("uint8")  #making sure all values are valid
    return compressed_image


def svd_compress(to_compress, k):
    U, S, Vt = np.linalg.svd(to_compress, full_matrices=False)   #svd decomposing
    U_k = U[:, :k]              #the k first left eigen vectors
    S_k = np.diag(S[:k])        #the k first singular values
    Vt_k = Vt[:k, :]            #the k first transposed right eigenvalues
    return U_k @ S_k @ Vt_k     #miltiplyng




def gray_compress(pic, k):
    img = Image.open(pic).convert("L")   #opening the image in grayscale mode
    matrix = np.array(img)
    compressed_pic = svd_compress(matrix, k)       #svd compressing the matrix(there is only 1 channel)
    compressed_pic = np.clip(compressed_pic, 0, 255).astype("uint8")       #insuring all values are valid
    return compressed_pic

def make_binary(pic):
    img = Image.open(pic).convert("L")    #converting to grayscale
    matrix = np.array(img)
    matrix = np.where(matrix > 128, 255, 0)  #if the color is brighter than 128, it becomes white, else, black.
    return matrix.astype('uint8')

def make_green(pic):    #keeping only the green components and throwing blue and red
    img = Image.open(pic)
    matrix = np.array(img)
    matrix[:, :, 0] = 0
    matrix[:, :, 2] = 0
    return matrix

def save_image(original_image, matrix, output_path="output_image.jpg"):
    img = Image.fromarray(matrix.astype('uint8'))     
    img.save(output_path)
    print(f"The image was saved as {output_path}")
    
    original_image_pil = Image.open(original_image)
    
    # Check if the image is grayscale or color
    if len(matrix.shape) == 2 or matrix.shape[2] == 1:  # It's a grayscale image
        img_cmap = "gray"  # Use gray colormap for grayscale images
    else:
        img_cmap = None  # Default for color images

    # Create the plot with 2 subplots for side-by-side display
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show the original image
    axes[0].imshow(original_image_pil, cmap=img_cmap)
    axes[0].set_title("Original Image")
    axes[0].axis("off")  # Hide axis for better viewing
    
    # Show the processed image
    axes[1].imshow(img, cmap=img_cmap)
    axes[1].set_title("Processed Image")
    axes[1].axis("off")  # Hide axis for better viewing
    
    # Display both images side by side
    plt.tight_layout()
    plt.show()


def make_red(pic):     #keeping only red components and throwing away the others
    img = Image.open(pic)
    matrix = np.array(img)
    matrix[:, :, 1] = 0
    matrix[:, :, 2] = 0
    return matrix



def make_blue(pic):    #keeping only blue components and throwing away the others
    img = Image.open(pic)
    matrix = np.array(img)
    matrix[:, :, 1] = 0
    matrix[:, :, 0] = 0
    return matrix


    

def apply_kernel(pic, kernel):
        img = Image.open(pic) 
        img = img.convert("RGBA")   #adding alpha channel for transperancy
        matrix = np.array(img)
        processed_image = np.zeros_like(matrix, dtype=np.float32)     #the matrix that will be the output
        processed_image[:,:,3] = matrix[:,:,3]     #copying the alpha channel
        height, width, channels = matrix.shape
        for i in range (1, height-1):
            for j in range (1, width-1):
                for c in range (3):
                    region = matrix[i-1:i+2, j-1:j+2]   #the 3x3 area of each spot in the matrix
                    processed_image[i, j, c] = np.sum(region[:,:,c]*kernel)    #summing the multyplication of each number in the area with it's equvilant from the kernel

        processed_image = np.clip(processed_image, 0, 255).astype('uint8')
        return processed_image
    
def save_processed_image(image_path, kernel, output_file="output_image.png"):
    # Apply the kernel to the image
    processed_image = apply_kernel(image_path, kernel)
    
    # Convert the processed NumPy array back to a PIL image
    processed_image_pil = Image.fromarray(processed_image)
    
    # Save as PNG to preserve transparency (alpha channel)
    processed_image_pil.save(output_file, "PNG")
    print(f"Image saved as {output_file}")


    original_image = Image.open(image_path)
    
    # Create the plot with 2 subplots for side-by-side display
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show the original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")  # Hide axis for better viewing
    
    # Show the processed image
    axes[1].imshow(processed_image_pil)
    axes[1].set_title("Processed Image")
    axes[1].axis("off")  # Hide axis for better viewing
    
    # Display both images side by side
    plt.tight_layout()
    plt.show()






def brighten_image(image, factor): #doubling each component by the factor
    img = Image.open(image)
    matrix = np.array(img)
    matrix = np.clip(matrix * factor, 0, 255)
    return matrix.astype(np.uint8)
       














def main():
   user_input = input("for compression, type 1. \nfor filters, type 2\n")
   if user_input != '1' and user_input != '2':
       print("invalid number")
       return


   if user_input == '1':
       user_input = input("for regular compression, type 1\nfor black and white compression, type 2\n")
       if user_input != '1' and user_input != '2':
            print("invalid number")
            return


       k = int(input("how much would you like to compress it? maximum: 1, minimum: 255\n*note: the bigger the compression, the blurrier the picture\n"))
       if k < 1 or k > 255:
           print("invalid number")
           return
       path = input("type the path to the picture you would like to compress\n")
       if not os.path.exists(path):
           print("the path does not exists")
           return
       

       if user_input == '1':
           save_image(path, color_compress(path, k))
       else:
           save_image(path, gray_compress(path, k))

   else:        
       filter = input("select your filter:\nblack and white: 1\nbinary black and white: 2\nblue: 3\ngreen: 4\nred: 5\nblurry: 6\nhigh color contrust: 7\nbright: 8\ndark: 9\n")
       
       valid_filters = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
       if filter not in valid_filters:
            print("Invalid filter selection")
            return 

       path = input("type the path to the picture\n") 
       if not os.path.exists(path):
           print("the path does not exists")
           return
       

       if filter == '1':
           save_image(path, gray_compress(path, 255))
       
       elif filter == '2':
           save_image(path, make_binary(path))

       elif filter == '3':
           save_image(path, make_blue(path)) 

       elif filter == '4':
           save_image(path, make_green(path))

       elif filter == '5':
           save_image(path, make_red(path)) 

       elif filter == '6':
           kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16
           save_processed_image(path, kernel)
      
       elif filter == '7': 
           kernel = np.array([[0, -0.1, 0], 
                       [-0.1, 2, -0.1], 
                       [0, -0.1, 0]], dtype=np.float32) 
           save_processed_image(path, kernel)

       elif filter == '8':
            factor = int(input("from 1 to 10, how much brighter would you like it?\n"))
            if (factor > 10 or factor < 1): 
                print("invalid number")
                return
            else:
                factor = factor/10
                save_image(path, brighten_image(path, 1 + factor))

       else:
               factor = int(input("from 1 to 5, how much darker would you like it?\n"))
               if (factor > 5 or factor < 1): 
                  print("invalid number")
                  return
               else:
                  factor = factor/10
                  save_image(path, brighten_image(path, 1 - factor))

   print("*note: the image will look better in the computer's image viewer") 

      
                   
               
           
             

           
   






main()