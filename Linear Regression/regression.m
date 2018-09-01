clc
data = load('group1_train.txt');

X= data(:,1);
y= data(:,3);
m = length(y);
theta = zeros(2,1);
i=1500;
alpha =0.01;

%Plot
plot(X,y,'rx','MarkerSize', 10);
title('Training Set');
xlabel('X');
ylabel('Y');
J=ComputeCost(X,y,theta);

[theta,j_history]= Grad(X,y,theta,alpha,i);
hold on;
plot(X(:,2),X*theta,'-');
legend('training data','Linear regression');
hold off;