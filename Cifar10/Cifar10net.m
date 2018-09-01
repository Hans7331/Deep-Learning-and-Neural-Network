cifar10Data = tempdir;
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
helperCIFAR10Data.download(url, cifar10Data);
[trainingImages, trainingLabels, testImages, testLabels] = helperCIFAR10Data.load(cifar10Data);



size(trainingImages)
numImageCategories = 10;
categories(trainingLabels)

figure
thumbnails = trainingImages(:,:,:,1:100);
thumbnails = imresize(thumbnails, [64 64]);
montage(thumbnails)


[height, width, numChannels, ~] = size(trainingImages);

imageSize = [height width numChannels];
inputLayer = imageInputLayer(imageSize)

filterSize = [5 5];
numFilters = 100;

middleLayers = [

convolution2dLayer(filterSize, numFilters, 'Padding', 2)

reluLayer()
maxPooling2dLayer(2, 'Stride', 1)


convolution2dLayer(filterSize, numFilters, 'Padding', 2)
reluLayer()
maxPooling2dLayer(2, 'Stride',1)

convolution2dLayer(filterSize, 2 * numFilters, 'Padding', 2)

reluLayer()
maxPooling2dLayer(2, 'Stride',1)

]

finalLayers = [


fullyConnectedLayer(64)
reluLayer
fullyConnectedLayer(numImageCategories)
softmaxLayer
classificationLayer
]

layers = [
    inputLayer
    middleLayers
    finalLayers
    ]

layers(2).Weights = 0.0001 * randn([filterSize numChannels numFilters]);

opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod', 10, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 50, ...
    'Verbose', true);

cifar10Net = trainNetwork(trainingImages, trainingLabels, layers, opts);


w = cifar10Net.Layers(8).Weights;
w = mat2gray(w);
w = imresize(w, 5);
figure 
montage(w)
figure 
montage(w(1:25,1:25,1:3,1:100))
class(w)
size(w)
YTest = classify(cifar10Net, testImages);
accuracy = sum(YTest == testLabels)/numel(testLabels)
%plotconfusion(testLabels,YTest)
%plotconfusion(YTestm,YTestm)
%plotconfusion(transpose(zeros(10,1)),transpose(ones(10,1)))

