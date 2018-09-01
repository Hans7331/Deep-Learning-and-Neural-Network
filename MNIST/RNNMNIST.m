%Loading Data

XTest = loadMNISTImages('t10k-images.idx3-ubyte');
YTest =loadMNISTLabels('t10k-labels.idx1-ubyte');
XTrain = loadMNISTImages('train-images.idx3-ubyte');
YTrain =loadMNISTLabels('train-labels.idx1-ubyte');

 save=zeros(25,28,28);
trainingImages= XTrain;
trainingLabels=YTrain;
testImages=XTest;
testLabels=YTest;
onett=zeros(28,28);
%Layer
perm = randperm(numel(YTrain), 25);

inputSize = 784;
numHiddenUnits = 100;
numClasses = 10;

layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits)

    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]

%Options

maxEpochs = 20;
miniBatchSize = 27;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate', 0.01, ...
    'Shuffle','never', ...
    'Verbose',1); 
    %'Plots','training-progress');
yt=categorical(YTrain);
yt=transpose(yt);


%Network Training

net = trainNetwork(XTrain,yt,layers,options);


miniBatchSize = 27;
YPred = classify(net,XTest, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');
YPred=double(YPred);
YPred=transpose(YPred);
for i=1:10000
    ytt(i)=YPred(i)-1;
end
ytt=transpose(ytt);
acc = sum(YPred == YTest)./numel(YTest)

acc = sum(ytt == YTest)./numel(YTest)
confusionmat(ytt,YTest)

