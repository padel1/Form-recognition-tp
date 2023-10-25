import numpy as np
import pygame
import cv2


class NeuralNetwork:
    learning_rate = 0.1

    def __init__(self, input_size, hidden_size, output_size):
        # Initialize the weights and biases
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def forward(self, X):
        # Perform the forward pass
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.softmax = np.exp(self.z2) / \
            np.sum(np.exp(self.z2), axis=1, keepdims=True)
 
        return self.softmax

    def backward(self, X, y, y_pred):
        # Compute the gradients
        self.dz2 = y_pred - y

        self.dw2 = np.dot(self.a1.T, self.dz2)
        self.db2 = np.sum(self.dz2, axis=0, keepdims=True)
        self.dz1 = np.dot(self.dz2, self.weights2.T) * \
            (1 - np.power(self.a1, 2))
        self.dw1 = np.dot(X.T, self.dz1)
        self.db1 = np.sum(self.dz1, axis=0, keepdims=True)

        # Update the weights and biases
        self.weights1 -= self.learning_rate * self.dw1
        self.bias1 -= self.learning_rate * self.db1
        self.weights2 -= self.learning_rate * self.dw2
        self.bias2 -= self.learning_rate * self.db2


    def train(self, X, y, epochs):
        # Train the network
        for epoch in range(epochs):
            y_pred = self.forward(X)
            # print("haa")
            if epoch %100==0 :
                loss = -np.mean(np.sum(y * np.log(y_pred), axis=1))
                print(f"Epoch {epoch}: loss={loss}")
                if loss<0.001:
                    break

            self.backward(X, y, y_pred)

            
    def predict(self, X):
    
        y_pred = self.forward(X)

        return np.argmax(y_pred, axis=1)


def count_zeros(image):
    cpt = 0
    for i in range(image.get_width()):
        for j in range(image.get_height()):
            if image.get_at((i, j))[0] == 0:
                cpt += 1
    return cpt


def split_image(image):
    width, height = pygame.math.Vector2(image.get_size())//2
    width = int(width)
    height = int(height)
    s1, s2, s3, s4 = pygame.Surface((width, height)), pygame.Surface(
        (width, height)), pygame.Surface((width, height)), pygame.Surface((width, height))
    for i in range(int(width)*2):
        for j in range(int(height)*2):
            if i < width:
                if j < height:
                    s1.set_at((i, j), image.get_at((i, j)))
                else:
                    s2.set_at((i, j-height), image.get_at((i, j)))
            else:
                if j < height:
                    s3.set_at((i-width, j), image.get_at((i, j)))
                else:
                    s4.set_at((i-width, j-height), image.get_at((i, j)))

    return (s1, s2, s3, s4)





# input_samples = []
# for i in range(10):
#     img = pygame.image.load(F"RF/split_digits/{i}.png")
#     img_split = split_image(img)
#     input_samples.append([count_zeros(img_split[0]), count_zeros(
#         img_split[1]), count_zeros(img_split[2]), count_zeros(img_split[3])])

# # print(input_samples)
# # for i in input_samples:
# #     print(i)
# input_size = 4
# hidden_size = 10
# output_size = 10

# nnk = NeuralNetwork(input_size, hidden_size, output_size)
# # # Convert the image to graysciale, resize it to 28x28 pixels and normalize it
# # img = pygame.surfarray.array3d(v)
# # img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])  # convert to grayscale
# # img = pygame.surfarray.make_surface(img).convert_alpha()
# # img = pygame.transform.scale(img, (28, 28))
# # img = pygame.surfarray.array2d(img)
# # img = img.astype('float32') / 255.

# x = np.array(input_samples)/np.max(input_samples)
# print(x)
# y = np.array([[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]])
# epochs = 200000
# # for i in x:
# #     print(i)
# nnk.train(x, y, epochs)
# # print("weights1",nnk.weights1,"\n")
# # print("weights2",nnk.weights2)
# # print("\nbias1",nnk.bias1)
# # print("\nbias2",nnk.bias2)
# while True:
#     inpute=int(input("give me a digit:"))
#     print(nnk.predict(x[inpute]))


