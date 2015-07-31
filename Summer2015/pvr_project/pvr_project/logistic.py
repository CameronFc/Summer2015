import numpy as np
import theano
import theano.tensor as T
import math
import header
rng = np.random

# OLD

class Logistic:
    def __init__(s, imageArray, classArray, nameList, steps = 100):
        s.steps = steps
        s.N = len(imageArray)
        s.nameList = nameList
        #print("length of image array ", len(imageArray))
        s.feats = len(imageArray[0])
        #print("length of images ", len(imageArray[0]))
        rca = []
        # for i in classArray:
        #     rca.append(i.get("objects").get("light").get("position"))
        # print(rca)
        for i in classArray:
            #print(i)
            rca.append(i.get("class"))
        #print(rca)
        s.D = (imageArray, rca)

        s.x = T.matrix("x")
        s.y = T.vector("y")

        s.w = theano.shared(rng.randn(s.feats), name="w")

        s.b = theano.shared(0., name="b")

        s.p_1 = 1 / (1 + T.exp(T.dot(s.x, s.w) + s.b))

        s.prediction = s.p_1 > 0.50

        s.xent = -s.y * T.log(s.p_1) - (1-s.y) * T.log(1-s.p_1)

        s.cost = s.xent.mean() + 0.01 * (s.w ** 2).sum()

        s.gw, s.gb = T.grad(s.cost, [s.w, s.b])

        s.train = theano.function(
              inputs=[s.x,s.y],
              outputs=[s.prediction, s.xent],
              #shift each of w,b by their respective gradients
              updates=((s.w, s.w - 0.1 * s.gw), (s.b, s.b - 0.1 * s.gb)))

        s.predict = theano.function(inputs=[s.x], outputs=s.prediction)


    def beginTraining(s):
        for i in range(s.steps):
            pred, err = s.train(s.D[0], s.D[1])
            if i + 1 == 1 or i + 1 == s.steps or (i + 1) % math.ceil(math.sqrt(s.steps)) == 0:
                print("Trained " + str(i + 1) + "/" + str(s.steps) + " steps")
        #print ("target values for D:", s.D[1])
        #print ("prediction on D:", s.predict(s.D[0]))
        #print ("Differences in target values and prediction:", s.D[1] - s.predict(s.D[0]))
        print("Correct classifications in training set: " + str(len(s.D[1]) - np.sum(np.abs(s.D[1] - s.predict(s.D[0])))) + "/" + str(len(s.D[1])))

    def classifyImages(s, imageArray, classArray, nameList):
        rca = []
        for i in classArray:
            rca.append(i.get("class"))
        dislikeCount = 0
        for i in range(len((imageArray))):
            image = imageArray[i]
            matrixFormatImage = image.reshape(1,s.feats)
            predictedClass = header.intToClass(1 if s.predict(matrixFormatImage) else 0)
            trueClass = header.intToClass(rca[i])
            #print(trueClass, predictedClass)
            if  predictedClass != trueClass:
                print("The " + str(trueClass) + " " + nameList[i] + " was misclassified as a " + predictedClass)
                dislikeCount += 1
        print("Misclassifications on test set: " + str(dislikeCount) + "/" + str(len(imageArray)))


    def classify(s, image, imageName):
        matrixFormatImage = image.reshape(1,s.feats)
        print("The object in " + imageName + " is 0 - sphere, 1 - cube: ", s.predict(matrixFormatImage))

# Train
#for i in range(training_steps):
#    pred, err = train(D[0], D[1])

#print ("Final model:")
#print (w.get_value(), b.get_value())
#print ("target values for D:", D[1])
#Now, use the cost-minimalized updated weights and base to generate results for each.
#print ("prediction on D:", predict(D[0]))
#How well did it work?
#print ("Differences in target values and prediction:", D[1] - predict(D[0]))