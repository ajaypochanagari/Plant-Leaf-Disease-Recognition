# Plant-Leaf-Disease-Recognition/
*This project is focused on building a machine learning model to recognize plant leaf diseases from images. The system can classify leaves into three categories:
---
1.Healthy
2.Powdery Mildew
3.Rust Disease
The model leverages Convolutional Neural Networks (CNNs) to perform image classification and achieve accurate predictions.
#Dataset
The dataset contains images of plant leaves divided into the following sets:
1.Training Set: Images used to train the model.
2.Validation Set: Images used to validate the model during training.
3.Test Set: Images used to evaluate the model's performance.
#Tools and Libraries:
The following tools and libraries were used:
->Python
->TensorFlow / Keras
->NumPy
->Pandas
->Matplotlib
->Seaborn
#Model Architecture
The Convolutional Neural Network consists of:
*Input Layer: Accepts RGB images of size (225x225x3).
*Convolutional Layers: Extracts features from the input images.
*MaxPooling Layers: Reduces the spatial dimensions.
*Flatten Layer: Converts the 2D features into a 1D array.
*Dense Layers: Performs classification into three categories using a softmax activation function.
*Hyperparameters:Optimizer: Adam
*Loss Function: Categorical Crossentropy
*Batch Size: 32
*Epochs: 20

##Steps to Run the Project:
1.Clone the repository:
git clone
2.Navigate to the project directory:
cd plant-leaf-disease-recognition
3.Install the required dependencies:
pip install -r requirements.txt
4.Test the model using sample images:
python predict.py --image_path /path/to/image.jpg

#ResultsTraining Accuracy: Achieved high accuracy on the training dataset.
->Validation Accuracy: Consistently high performance during validation.
->The model can classify images into the correct category with significant accuracy, making it a useful![Screenshot 2024-11-17 223322](https://github.com/user-attachments/assets/1c4629e6-b4f4-4a6d-b076-66d8012d65e2)
 tool for early disease detection in plants.



![Screenshot 2024-11-28 195529](https://github.com/user-attachments/assets/ab38f4b4-e830-40f5-8110-ea2a5386e34e)




![Screenshot 2024-11-28 195621](https://github.com/user-attachments/assets/c879a721-9608-4d52-b90c-cf79f4976648)
