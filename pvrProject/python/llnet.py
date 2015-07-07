import numpy as np
import theano
import theano.tensor as T
import math
import header
rng = np.random

#Logistic-linear net to estimate the position of lights
#TODO: change to lights not cube sphere
#TODO: create feature to store the weights of a learned mlp for storage and later use

#theano.config.compute_test_value = 'warn'

class HiddenLayer:
    def __init__(s, input, dims, activation=None):
        wBase = rng.randn(*dims)
        s.w = theano.shared(wBase, name='w')
        s.b = theano.shared(0. , name='b')
        linOutput = T.dot(input, s.w) + s.b
        s.output = (
            linOutput if activation is None
            else activation(linOutput)
        )
        s.params = [s.w,s.b]

class LLnet:
    def __init__(s, imageArray, classArray, nameList, steps = 20):
        L1reg = 0.0001
        s.steps = steps
        s.N = len(imageArray)
        print(imageArray)
        #number of neuron stacks in logistic layer
        s.f2 = 10
        s.nameList = nameList
        print("length of image array ", len(imageArray))
        s.feats = len(imageArray[0])
        print("length of images ", len(imageArray[0]))
        rca = []
        for i in classArray:
            rca.append(i.get("objects").get("light").get("position"))
        print(rca)
        s.D = (imageArray, rca)
        #s.D = (imageArray, np.ones((100,3)) * 0.00001)

        s.x = T.matrix("x")
        s.y = T.matrix("y")

        hl = HiddenLayer(s.x, [s.feats, s.f2], T.tanh)
        ll = HiddenLayer(hl.output, [s.f2, 3])
        s.params = hl.params + ll.params

        #How many weights in the logistic layer? there should be a total of n * 3 neurons here, each with s.feats parameters
        #s.w = theano.shared(rng.randn(s.feats, s.f2), name="w")
        #In the linear layer, there should be a total of n
        #s.w2 = theano.shared(rng.randn(s.f2, 3), name="w2")

        #s.b = theano.shared(0., name="b")
        #s.b2 = theano.shared(0., name="b2")

        #s.p_1 = 1 / (1 + T.exp(T.dot(s.x, s.w) + s.b))

        #s.l = T.dot(s.p_1, s.w2) + s.b2


        s.regularization = 0.0
        for i in (param.sum() for param in s.params):
            s.regularization += L1reg * i

        s.cost = ((ll.output - s.y)**2).sum() + s.regularization

        s.prediction = theano.function(inputs=[s.x], outputs=[ll.output])

        s.getCost = theano.function(inputs=[s.x, s.y], outputs=[s.cost])

        #s.gw, s.gw2, s.gb, s.gb2 = T.grad(s.cost, [s.w, s.w2, s.b, s.b2])

        s.gparams = [T.grad(s.cost, param) for param in s.params]

        s.learningRate = 0.001

        s.updates = [
            (param, param - s.learningRate * gparam) for param, gparam in zip(s.params,s.gparams)
        ]

        s.train = theano.function(
              inputs=[s.x,s.y],
              outputs=[s.cost],
              #shift each of w,b by their respective gradients
              #updates=((s.w, s.w - 0.1 * s.gw), (s.w2, s.w2 - 0.001 * s.gw2), (s.b, s.b - 0.1 * s.gb), (s.b2, s.b2 - 0.001 * s.gb2)))
              updates=s.updates
        )
        #s.predict = theano.function(inputs=[s.x], outputs=s.prediction)

    def beginTraining(s):
        #print(len(s.D[1]), len(s.D[1][0]))
        #theano.printing.debugprint(s.cost)
        oldCost = 0.0
        for i in range(s.steps):
            currentCost = s.train(s.D[0], s.D[1])[0]
            #Print loop
            if i + 1 == 1 or i + 1 == s.steps or (i + 1) % math.ceil(math.sqrt(s.steps)) == 0:
                print("Trained {}/{} steps, cost : {} improvement: +{:%}".format(str(i + 1),str(s.steps),currentCost, (oldCost - currentCost)/currentCost))
                oldCost = currentCost
        #print ("target values for D:", s.D[1])
        #print ("prediction on D:", s.predict(s.D[0]))
        #print ("Differences in target values and prediction:", s.D[1] - s.predict(s.D[0]))
        #print("Correct classifications in training set: " + str(len(s.D[1]) - np.sum(np.abs(s.D[1] - s.predict(s.D[0])))) + "/" + str(len(s.D[1])))

    def classify(s, images):
        #matrixFormatImage = image.reshape(1,s.feats)
        #print("Light position: ", s.prediction(matrixFormatImage))
        #print(s.w.get_value(), s.b.get_value(), s.w2.get_value(), s.b2.get_value())
        print(s.getCost(s.D[0], s.D[1]))

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