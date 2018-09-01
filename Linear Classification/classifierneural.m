
load('data.mat');
load('val.mat');
load('test.mat');

XTrain=transpose(data(:,(1:2)));
XTest=transpose(test(:,(1:2)));

%plot(transpose(YPred),transpose(YTest),'rx')
plot(XTest(1,:),XTest(2,:),'rx')
for i=1:1000
    if(data(i,3)==1)
        YTrain(i)=0;
    end
    if(data(i,4)==1)
         YTrain(i)=1;
        end
    if(data(i,5)==1)
         YTrain(i)=2;
        end
    if(data(i,6)==1)
         YTrain(i)=3;
        end
end        


for i=1:400
    if(test(i,3)==1)
        YTest(i)=0;
    end
    if(test(i,4)==1)
         YTest(i)=1;
        end
    if(test(i,5)==1)
         YTest(i)=2;
        end
    if(test(i,6)==1)
         YTest(i)=3;
        end
end  

%{
for i=1:600
    if(val(i,3)==1)
        YTest(i)=0;
    end
    if(val(i,4)==1)
         YTest(i)=1;
        end
    if(val(i,5)==1)
         YTest(i)=2;
        end
    if(val(i,6)==1)
         YTest(i)=3;
        end
end  
%}
x1=XTrain(1,:);
y1=XTrain(2,:);

x2=x1;
y2=y1;
plot(x2,y2,'rx')
figure
plot(x2(:,1:250),y2(:,1:250),'rx')
figure
plot(x2(:,251:500),y2(:,251:500),'rx')
figure
plot(x2(:,501:750),y2(:,501:750),'rx')
figure
plot(x2(:,751:1000),y2(:,751:1000),'rx')

XTrain(1,:)=x2;
XTrain(2,:)=y2;
plot(XTrain(1,:),XTrain(2,:),'rx')




%////////////////////////////////////////////////////////

targetValues = 0.*ones(10, size(YTrain, 1));
    for n = 1: size(YTrain, 1)
        targetValues(YTrain(n) + 1, n) = 1;
    end;


    
    
batchSize = 150;
epochs =700;
numberOfHiddenUnits = 900;
learningRate = 0.01;  


activationFunction = @logisticSigmoid;
dActivationFunction = @dLogisticSigmoid;

%///////////
 trainingSetSize = size(XTrain,2);
    
    % Input vector has 784 dimensions.
    inputDimensions = size(XTrain, 1);
    % We have to distinguish 10 digits.
    outputDimensions = size(targetValues, 1);
    
    % Initialize the weights for the hidden layer and the output layer.
    hiddenWeights = rand(numberOfHiddenUnits, inputDimensions);
    outputWeights = rand(outputDimensions, numberOfHiddenUnits);
    
    hiddenWeights = hiddenWeights./size(hiddenWeights, 2);
    outputWeights = outputWeights./size(outputWeights, 2);
    
    n = zeros(batchSize);
    
    figure; hold on;

    for t = 1: epochs %Epochs
        for k = 1: batchSize
            % Select which input vector to train on.
            n(k) = floor(rand(1)*trainingSetSize + 1);
            
            % Propagate the input vector through the network.
            inputVector = XTrain(:, n(k));
            hiddenActualInput = hiddenWeights*inputVector;
            hiddenOutputVector = activationFunction(hiddenActualInput);
            outputActualInput = outputWeights*hiddenOutputVector;
            outputVector = activationFunction(outputActualInput);
            
            targetVector = targetValues(:, n(k));
            
            % Backpropagate the errors.
            outputDelta = dActivationFunction(outputActualInput).*(outputVector - targetVector);
            hiddenDelta = dActivationFunction(hiddenActualInput).*(outputWeights'*outputDelta);
            
            outputWeights = outputWeights - learningRate.*outputDelta*hiddenOutputVector';
            hiddenWeights = hiddenWeights - learningRate.*hiddenDelta*inputVector';
        end;
        
        % Calculate the error for plotting.
        error = 0;
        for k = 1: batchSize
            inputVector = XTrain(:, n(k));
            targetVector = targetValues(:, n(k));
            
            error = error + norm(activationFunction(outputWeights*activationFunction(hiddenWeights*inputVector)) - targetVector, 2);
        end;
        error = error/batchSize;
        
        plot(t, error,'*');
    end;
    q=1;
   YPred=zeros(1,600); 
    %/////////////
testSetSize = size(XTest, 2);
    classificationErrors = 0;
    correctlyClassified = 0;
    
    for n = 1: testSetSize
        inputVector = XTest(:, n);
        outputVector = activationFunction(outputWeights*activationFunction(hiddenWeights*inputVector));
        
        max = 0;
    class = 1;
    for i = 1: size(outputVector, 1)
        if outputVector(i) > max
            max = outputVector(i);
            class = i;
            
        end;
    end;
         
        if class == YTest(n) + 1
            correctlyClassified = correctlyClassified + 1;
        else
            classificationErrors = classificationErrors + 1;
        end;
        YPred(q)=class-1;
            q=q+1;
    end;
acc = correctlyClassified /(correctlyClassified + classificationErrors)
confusionmat(YTest,YPred)