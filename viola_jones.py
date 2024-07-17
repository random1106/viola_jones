import numpy as np
import math

def integral_image(image):

    """Computes the sum of image data in rectange 
    [0, x] * [0, y] 
    
    Paraemeters
    -------------
    image: np.array 
        m * n matrix which contains image pixel

    Returns
    -------------
    integral_image: np.array
        m * n which cotains the sum of image pixel in 
            rectange [0, x] * [0, y] 
    """

    height, width = image.shape
    s = np.zeros((height + 1, width + 1))
    integral = np.zeros((height + 1, width + 1))

    for i in range(height + 1):
        for j in range(width + 1):
            if j == 0:
                s[i][j] = 0
            else:
                s[i][j] = s[i][j-1] + image[i-1][j-1]
            
            if i == 0:
                integral[i][j] = 0
            else:
                integral[i][j] = integral[i-1][j] + s[i][j]
    
    return integral

class RectangleRegion:
    
    """
        Class that stores the rectange region [x, x + width) * [y, y + height)
    """
    
    def __init__(self, x, y, height, width):
        self.x = x
        self.y = y
        self.height = height
        self.width  = width

    def compute_integral(self, integral):
        
        """
        function that computes the total sum of pixels
        """
        
        return integral[self.x + self.height][self.y + self.width] + integral[self.x][self.y] - \
               integral[self.x][self.y + self.width] - integral[self.x + self.height][self.y]

class WeakClassifier:
    
    def __init__(self, positive_regions, negative_regions, threshold, polarity):
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions
        self.threshold = threshold
        self.polarity = polarity

    def classify(self, x):
        
        """
        Parameters
        ------------
        x: integral_image of an image

        Returns
        ------------
        classification, either 1 or 0
        
        """

        feature = lambda ii: sum([pos.compute_integral(ii) for pos in self.positive_regions]) - \
                             sum([neg.compute_integral(ii) for neg in self.negative_regions])
        return 1 if self.polarity * feature(x) < self.polarity * self.threshold else 0

class ViolaJones:
    def __init__(self, T = 10):
        self.T = T
        self.alphas = []
        # list of weak classifiers
        self.clfs = []

    def build_features(self, image_shape):
        
        """
        Returns
        ---------
        a list of [positive_regions, negative_regions]
        where positive_regions and negative_regions themselves are list of RectangleRegion
        given in white and black in Figure 1 of viola_jones_01.pdf 
        """


        print("build_features starts")

        # here, the image_shape is the shape of integral_image

        height = image_shape[0] - 1 
        width = image_shape[1] - 1
        
        features = []

        for x in range(height):
            for y in range(width):
                for h in range(1, height + 1):
                    for w in range(1, width + 1):
                        # type A region
                        if x + h <= height and  y + 2 * w <= width:
                            positive_regions = [RectangleRegion(x, y, h, w)]
                            negative_regions = [RectangleRegion(x, y + w, h, w)]
                            features.append([positive_regions, negative_regions])
                        if x + 2 * h <= height and y + w <= width:
                            positive_regions = [RectangleRegion(x + h, y, h, w)]
                            negative_regions = [RectangleRegion(x, y, h, w)]
                            features.append([positive_regions, negative_regions])
                        if x + h <= height and y + 3 * w <= width:
                            positive_regions = [RectangleRegion(x, y, h, w), \
                                                RectangleRegion(x, y + 2 * w, h, w)]
                            negative_regions = [RectangleRegion(x, y + w, h, w)]
                            features.append([positive_regions, negative_regions])
                        if x + 3 * h <= height and y + w <= width:
                            positive_regions = [RectangleRegion(x, y, h, w), \
                                                RectangleRegion(x + 2 * h, y, h, w)]
                            negative_regions = [RectangleRegion(x + h, y, h, w)]
                            features.append([positive_regions, negative_regions])
                        if x + 2 * h <= height and y + 2 * w <= width:
                            positive_regions = [RectangleRegion(x, y, h, w), \
                                                RectangleRegion(x + h, y + w, h, w)]
                            negative_regions = [RectangleRegion(x + h, y, h, w), \
                                                RectangleRegion(x, y + w, h, w)]
                            features.append([positive_regions, negative_regions])

        print("build_features ends")
                            
        return features
    
    def apply_features(self, features, training_data):
        
        """
        Parameters
        ------------
        training_data: a list [data, ...] where data = [data[0], data[1]], 
        data[0] is the INTEGRAL MATRIX of the pixel, and data[1] is the classification
        
        Returns
        ------------
        X: feature matrix: X[i] holds the features computed from training_data[i]
        y: classification
        """

        print("apply_features starts")

        X = np.zeros((len(training_data), len(features)))
        y = np.zeros(len(training_data))

        for i in range(len(training_data)):
            y[i] = training_data[i][1]
        
        
        for j in range(len(features)):
            positive_regions = features[j][0]
            negative_regions = features[j][1]
            feature = lambda ii: sum([pos.compute_integral(ii) for pos in positive_regions]) - \
                                 sum([neg.compute_integral(ii) for neg in negative_regions])
            for i in range(len(training_data)):
                X[i][j] = feature(training_data[i][0])

        print("apply_features ends")
        
        return X, y
    
    def train(self, trainings, pos_num, neg_num):
        
        """
        Parameters
        ------------
        trainings: a list [data...] where data[0] is the matrix of image pixel,
        data[1] is classification
        """
        
        training_data = []
        weights = np.zeros(len(trainings))
        for x in range(len(trainings)):
            training_data.append((integral_image(trainings[x][0]), trainings[x][1]))
            
            if trainings[x][1] == 1:
                weights[x] = 1 / (2 * pos_num)
            else:
                weights[x] = 1 / (2 * neg_num)
            
        features = self.build_features(training_data[0][0].shape)
        X, y = self.apply_features(features, training_data)

        for t in range(self.T):
            print(f"training step t = {t}")
            weights = weights / np.sum(weights)
            weak_classifiers = self.train_weak(X, y, features, weights)
            clf, error, accuracies = self.select_best(weak_classifiers, weights, training_data)
            beta = error / (1.0 - error)
            for i in range(len(weights)): 
                weights[i] = weights[i] * (beta ** (1 - accuracies[i]))

            self.alphas.append(math.log(1.0 / beta))
            self.clfs.append(clf)

    def train_weak(self, X, y, features, weights):
        weak_classifiers = []
        total_features = X.shape[1]

        for index, feature in enumerate(X.T):
            if len(weak_classifiers) % 1000 == 0:
                print(f"Trained {len(weak_classifiers)} classifiers out of {total_features}")
            
            error, threshold, polarity = float("inf"), None, None  
            combined = sorted(zip(feature, weights, y), key = lambda x: x[0])
            current_error_1 = 0
            current_error_2 = 0

            for i in range(len(combined)):
                current_error_1 += combined[i][2] * combined[i][1]
                current_error_2 += (1 - combined[i][2]) * combined[i][1]
            if current_error_1 <= current_error_2:
                error = min(current_error_1, current_error_2)
                threshold = combined[0][0]
                polarity = 1
            else:
                error = min(current_error_1, current_error_2)
                threshold = combined[0][0]
                polarity = -1

            for i in range(1, len(combined)):
                if combined[i-1][2] == 1:
                    current_error_1 -= combined[i-1][1]
                    current_error_2 += combined[i-1][1]
                else:
                    current_error_1 += combined[i-1][1]
                    current_error_2 -= combined[i-1][1]
                if min(current_error_1, current_error_2) < error :
                    if current_error_1 <= current_error_2:
                        error = min(current_error_1, current_error_2)
                        threshold = combined[i][0]
                        polarity = 1
                    else:
                        error = min(current_error_1, current_error_2)
                        threshold = combined[i][0]
                        polarity = -1
                                  
            positive_regions = features[index][0] 
            negative_regions = features[index][1]
            weak_classifiers.append(WeakClassifier(positive_regions, negative_regions, \
                                    threshold, polarity))           
        
        return weak_classifiers

    def select_best(self, classifiers, weights, training_data):
        best_clf, best_error, best_accuracies = None, float("inf"), None
        for clf in classifiers:
            error, accurcies = 0, []
            for w, data in zip(weights, training_data):
                correctness = abs(clf.classify(data[0]) - data[1])
                accurcies.append(correctness)
                error += w * correctness
            
            if error < best_error:
                best_error, best_clf, best_accuracies = error, clf, accurcies
                
        return best_clf, best_error, best_accuracies
    
    def classify(self, image):
        total = 0

        for t in range(self.T):
            total += self.alphas[t] * self.clfs[t].classify(integral_image(image)) 

        return 1 if total >=  1.0 / 2 * sum(self.alphas) else 0