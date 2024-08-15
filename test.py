import tensorflow as tf
import numpy as np
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

# Load the trained model from the .keras file
model = tf.keras.models.load_model('C:/Users/rishi/Desktop/cvArVr/Project-Based-Computer-Vision-Learning/final_model_for_nuclei.keras')

# Function to preprocess the input image
def preprocess_image(image_path, img_height=128, img_width=128):
    # Load the image
    image = imread(image_path)
    original_shape = image.shape

    # Resize the image to the model's input size
    image_resized = resize(image, (img_height, img_width), mode='constant', preserve_range=True)
    image_resized = image_resized.astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    # Add a batch dimension
    image_input = np.expand_dims(image_resized, axis=0)
    
    return image_input, original_shape

# Function to predict and display the mask
def predict_mask(image_path, model, img_height=128, img_width=128):
    # Preprocess the input image
    image_input, original_shape = preprocess_image(image_path, img_height, img_width)

    # Use the model to predict the mask
    predicted_mask = model.predict(image_input, verbose=1)

    # Threshold the predicted mask to create a binary mask
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

    # Resize the mask to the original image size
    predicted_mask_resized = resize(np.squeeze(predicted_mask), (original_shape[0], original_shape[1]), 
                                    mode='constant', preserve_range=True)

    # Display the original image and the predicted mask
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    imshow(imread(image_path))
    plt.title('Original Image')
    
    plt.subplot(1, 2, 2)
    imshow(predicted_mask_resized, cmap='gray')
    plt.title('Predicted Mask')
    
    plt.show()

# Example usage
image_path = 'C:/Users/rishi/Desktop/cvArVr/Project-Based-Computer-Vision-Learning/stage1_test/0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732/images/0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732.png'
predict_mask(image_path, model)
