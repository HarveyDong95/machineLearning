{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "from tqdm import tqdm_notebook\n",
    "import pandas as pd\n",
    "from keras.utils import np_utils\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Logic Based FizzBuzz Function [Software 1.0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "def fizzbuzz(n):\n",
    "    \n",
    "    # Logic Explanation\n",
    "    if n % 3 == 0 and n % 5 == 0:\n",
    "        return 'FizzBuzz'\n",
    "    elif n % 3 == 0:\n",
    "        return 'Fizz'\n",
    "    elif n % 5 == 0:\n",
    "        return 'Buzz'\n",
    "    else:\n",
    "        return 'Other'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Create Training and Testing Datasets in CSV Format"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "def createInputCSV(start,end,filename):\n",
    "    \n",
    "    # Why list in Python?\n",
    "    #\n",
    "    inputData   = []\n",
    "    outputData  = []\n",
    "    \n",
    "    # Why do we need training Data?\n",
    "    for i in range(start,end):\n",
    "        inputData.append(i)\n",
    "        outputData.append(fizzbuzz(i))\n",
    "    \n",
    "    # Why Dataframe?\n",
    "    dataset = {}\n",
    "    dataset[\"input\"]  = inputData\n",
    "    dataset[\"label\"] = outputData\n",
    "    \n",
    "    # Writing to csv\n",
    "    pd.DataFrame(dataset).to_csv(filename)\n",
    "    \n",
    "    print(filename, \"Created!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Processing Input and Label Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "def processData(dataset):\n",
    "    \n",
    "    # Why do we have to process?\n",
    "    data   = dataset['input'].values\n",
    "    labels = dataset['label'].values\n",
    "    \n",
    "    processedData  = encodeData(data)\n",
    "    processedLabel = encodeLabel(labels)\n",
    "    \n",
    "    return processedData, processedLabel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "def encodeData(data):\n",
    "    \n",
    "    processedData = []\n",
    "    \n",
    "    for dataInstance in data:\n",
    "        \n",
    "        # Why do we have number 10?\n",
    "        processedData.append([dataInstance >> d & 1 for d in range(10)])\n",
    "    \n",
    "    return np.array(processedData)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "def encodeLabel(labels):\n",
    "    \n",
    "    processedLabel = []\n",
    "    \n",
    "    for labelInstance in labels:\n",
    "        if(labelInstance == \"FizzBuzz\"):\n",
    "            # Fizzbuzz\n",
    "            processedLabel.append([3])\n",
    "        elif(labelInstance == \"Fizz\"):\n",
    "            # Fizz\n",
    "            processedLabel.append([1])\n",
    "        elif(labelInstance == \"Buzz\"):\n",
    "            # Buzz\n",
    "            processedLabel.append([2])\n",
    "        else:\n",
    "            # Other\n",
    "            processedLabel.append([0])\n",
    "\n",
    "    return np_utils.to_categorical(np.array(processedLabel),4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "training.csv Created!\n",
      "testing.csv Created!\n"
     ]
    }
   ],
   "source": [
    "# Create datafiles\n",
    "createInputCSV(101,1001,'training.csv')\n",
    "createInputCSV(1,101,'testing.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Read Dataset\n",
    "trainingData = pd.read_csv('training.csv')\n",
    "testingData  = pd.read_csv('testing.csv')\n",
    "\n",
    "# Process Dataset\n",
    "processedTrainingData, processedTrainingLabel = processData(trainingData)\n",
    "processedTestingData, processedTestingLabel   = processData(testingData)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Tensorflow Model Definition"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Defining Placeholder\n",
    "inputTensor  = tf.placeholder(tf.float32, [None, 10])\n",
    "outputTensor = tf.placeholder(tf.float32, [None, 4])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "NUM_HIDDEN_NEURONS_LAYER_1 = 200      ###edited\n",
    "LEARNING_RATE = 0.05                  ###edited\n",
    "\n",
    "# Initializing the weights to Normal Distribution\n",
    "def init_weights(shape):\n",
    "    return tf.Variable(tf.random_normal(shape,stddev=0.01))\n",
    "\n",
    "# Initializing the input to hidden layer weights\n",
    "input_hidden_weights  = init_weights([10, NUM_HIDDEN_NEURONS_LAYER_1])\n",
    "# Initializing the hidden to output layer weights\n",
    "hidden_output_weights = init_weights([NUM_HIDDEN_NEURONS_LAYER_1, 4])\n",
    "\n",
    "# Computing values at the hidden layer\n",
    "hidden_layer = tf.nn.relu6(tf.matmul(inputTensor, input_hidden_weights))\n",
    "# Computing values at the output layer\n",
    "output_layer = tf.matmul(hidden_layer, hidden_output_weights)\n",
    "\n",
    "# Defining Error Function\n",
    "error_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=outputTensor))\n",
    "\n",
    "# Defining Learning Algorithm and Training Parameters  ###edited\n",
    "training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(error_function)\n",
    "\n",
    "# Prediction Function\n",
    "prediction = tf.argmax(output_layer, 1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Training the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d52f1dd195c64a2d9dce0efd32877d01",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "HBox(children=(IntProgress(value=0, max=5000), HTML(value='')))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "NUM_OF_EPOCHS = 5000\n",
    "BATCH_SIZE = 128\n",
    "\n",
    "training_accuracy = []\n",
    "\n",
    "with tf.Session() as sess:\n",
    "    \n",
    "    # Set Global Variables ?\n",
    "    tf.global_variables_initializer().run()\n",
    "    \n",
    "    for epoch in tqdm_notebook(range(NUM_OF_EPOCHS)):\n",
    "        \n",
    "        #Shuffle the Training Dataset at each epoch\n",
    "        p = np.random.permutation(range(len(processedTrainingData)))\n",
    "        processedTrainingData  = processedTrainingData[p]\n",
    "        processedTrainingLabel = processedTrainingLabel[p]\n",
    "        \n",
    "        # Start batch training\n",
    "        for start in range(0, len(processedTrainingData), BATCH_SIZE):\n",
    "            end = start + BATCH_SIZE\n",
    "            sess.run(training, feed_dict={inputTensor: processedTrainingData[start:end], \n",
    "                                          outputTensor: processedTrainingLabel[start:end]})\n",
    "        # Training accuracy for an epoch\n",
    "        training_accuracy.append(np.mean(np.argmax(processedTrainingLabel, axis=1) ==\n",
    "                             sess.run(prediction, feed_dict={inputTensor: processedTrainingData,\n",
    "                                                             outputTensor: processedTrainingLabel})))\n",
    "    # Testing\n",
    "    predictedTestLabel = sess.run(prediction, feed_dict={inputTensor: processedTestingData})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x26e1e9ee9e8>"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df = pd.DataFrame()\n",
    "df['acc'] = training_accuracy\n",
    "df.plot(grid=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "def decodeLabel(encodedLabel):\n",
    "    if encodedLabel == 0:\n",
    "        return \"Other\"\n",
    "    elif encodedLabel == 1:\n",
    "        return \"Fizz\"\n",
    "    elif encodedLabel == 2:\n",
    "        return \"Buzz\"\n",
    "    elif encodedLabel == 3:\n",
    "        return \"FizzBuzz\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Testing the Model [Software 2.0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Errors: 2  Correct :98\n",
      "Testing Accuracy: 98.0\n"
     ]
    }
   ],
   "source": [
    "wrong   = 0\n",
    "right   = 0\n",
    "\n",
    "predictedTestLabelList = []\n",
    "\n",
    "for i,j in zip(processedTestingLabel,predictedTestLabel):\n",
    "    predictedTestLabelList.append(decodeLabel(j))\n",
    "    \n",
    "    if np.argmax(i) == j:\n",
    "        right = right + 1\n",
    "    else:\n",
    "        wrong = wrong + 1\n",
    "\n",
    "print(\"Errors: \" + str(wrong), \" Correct :\" + str(right))\n",
    "\n",
    "print(\"Testing Accuracy: \" + str(right/(right+wrong)*100))\n",
    "\n",
    "# Please input your UBID and personNumber \n",
    "testDataInput = testingData['input'].tolist()\n",
    "testDataLabel = testingData['label'].tolist()\n",
    "\n",
    "testDataInput.insert(0, \"UBID\")\n",
    "testDataLabel.insert(0, \"50291149\")\n",
    "\n",
    "testDataInput.insert(1, \"personNumber\")\n",
    "testDataLabel.insert(1, \"50291149\")\n",
    "\n",
    "predictedTestLabelList.insert(0, \"\")\n",
    "predictedTestLabelList.insert(1, \"\")\n",
    "\n",
    "output = {}\n",
    "output[\"input\"] = testDataInput\n",
    "output[\"label\"] = testDataLabel\n",
    "\n",
    "output[\"predicted_label\"] = predictedTestLabelList\n",
    "\n",
    "opdf = pd.DataFrame(output)\n",
    "opdf.to_csv('output.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
