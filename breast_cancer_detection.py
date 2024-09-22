import cv2
import matplotlib.pyplot as plt


def load_and_display_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converte BGR para RGB
    plt.imshow(image)
    plt.axis('off')
    plt.title('Imagem Original')
    plt.show()
    return image
def separate_color_channels(image):
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(red_channel, cmap='Reds')
    plt.title('Canal Vermelho')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(green_channel, cmap='Greens')
    plt.title('Canal Verde')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(blue_channel, cmap='Blues')
    plt.title('Canal Azul')
    plt.axis('off')

    plt.show()
    return red_channel, green_channel, blue_channel

def preprocess_image(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    plt.imshow(blurred_image)
    plt.title('Imagem Pré-processada')
    plt.axis('off')
    plt.show()
    return blurred_image

def analyze_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

    plt.imshow(binary_image, cmap='gray')
    plt.title('Resultado da Análise')
    plt.axis('off')
    plt.show()
    
    num_white_pixels = cv2.countNonZero(binary_image)
    num_black_pixels = binary_image.size - num_white_pixels

    presence_of_cancer = 'Possivelmente presença de câncer' if num_white_pixels > 1000 else 'Possivelmente ausência de câncer'
    
    return presence_of_cancer, num_white_pixels, num_black_pixels

image_path = 'C:/Users/CarlosVitor/Cancerdemama/mama.jpg'  
image = load_and_display_image(image_path)

red_channel, green_channel, blue_channel = separate_color_channels(image)

preprocessed_image = preprocess_image(image)

result, num_white_pixels, num_black_pixels = analyze_image(preprocessed_image)

print(result)
print(f"Número de pixels brancos: {num_white_pixels}")
print(f"Número de pixels pretos: {num_black_pixels}")
