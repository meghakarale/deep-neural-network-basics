
# Deep Neural Network Tutorial – Detailed Explanation

This tutorial introduces Deep Neural Networks (DNNs) to beginners by walking through the process of building and training a model to classify handwritten digits using the MNIST dataset with TensorFlow and Keras.

## What is a Deep Neural Network?

A Deep Neural Network is a type of machine learning model inspired by the human brain. It consists of layers of interconnected neurons (nodes). These layers are:
- Input Layer – receives the input data
- Hidden Layers – extract patterns through weighted computations
- Output Layer – produces the final prediction (e.g., digit class)

## Step 1: Load the MNIST Dataset

MNIST is a standard benchmark dataset containing 70,000 grayscale images of handwritten digits (0–9). It is split into:
- 60,000 training samples
- 10,000 test samples

We use TensorFlow’s built-in loader to retrieve the dataset.

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

## Step 2: Visualize the Data

Before modeling, we display one sample to understand the data format:
- Each image is 28x28 pixels.
- Pixel values range from 0 to 255 (brightness).

```python
plt.imshow(x_train[0], cmap='gray')
```

## Step 3: Preprocess the Data

Neural networks perform better when input values are normalized and labels are encoded:
- Normalize pixel values to the range [0, 1]
- Convert labels to one-hot vectors (e.g., 5 → [0,0,0,0,0,1,0,0,0,0])

```python
x_train = x_train / 255.0
y_train = to_categorical(y_train, 10)
```

## Step 4: Build the Neural Network

We use Keras's Sequential API to define a simple DNN:
- Flatten: Converts 2D images (28x28) into 1D arrays
- Dense(128, relu): First hidden layer with 128 neurons and ReLU activation
- Dense(64, relu): Second hidden layer
- Dense(10, softmax): Output layer for 10 digit classes

```python
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```

## Step 5: Compile and Train the Model

We compile the model using:
- Loss: categorical_crossentropy (for multi-class classification)
- Optimizer: adam (adaptive gradient-based optimizer)
- Metric: accuracy

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
```

During training:
- The model learns by adjusting weights to minimize loss
- Training occurs over multiple epochs (passes over the dataset)

## Step 6: Evaluate the Model

We test the model on unseen data (test set) to measure its performance.

```python
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

This gives us an unbiased estimate of the model’s generalization ability.

## Step 7: Make Predictions

We generate predictions for test images and compare them with actual labels:

```python
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)
```

This step simulates real-world usage—taking new input and predicting a result.

## Summary

By the end of this tutorial, you learned how to:
- Understand the structure and components of a DNN
- Preprocess image data for input to neural networks
- Build a multi-layer perceptron model using Keras
- Train the model on labeled data and evaluate its accuracy
- Make predictions using the trained model

This example is a perfect starting point to explore deeper concepts such as:
- Convolutional Neural Networks (CNNs)
- Hyperparameter tuning
- Dropout and regularization
- Saving/loading models
- Deployment via Streamlit or Flask
