function [MSE,alloutputs] = experiment1(neuron,train_algorithms,activation_func,input_values,target_values)

%----------------------------------------------------------------
% This script is a single layer feed-forward network and
% takes 5 parameters, which is 
% a number of neuron, training algorihm, activation function,
% input values and target values

% It returns final MSE value and all MSE outputs.
% You can see also training, validation and test values
% and the plots are at the bottom of the code. 

%----------------------------------------------------------------
%--------------------------------------------------------------------

net = feedforwardnet(neuron); % one hidden layer, 2,64,128
%You can use fitnet([neuron],trainFcn) instead of feedforwardnet

net.trainFcn = train_algorithms; 

%You can use fitnet([neuron],trainFcn) instead of feedforwardnet

% trainlm, Levenberg-Marquardt back-propagation 
% trainscg, Scaled conjugate gradient back-propagation
% traingdm, Gradient descent with momentum
% traingda, Gradient descent with adaptive learning rate backpropagation
% traingdx, Gradient descent with momentum and adaptive learning rate backpropagation
% traingd, Gradient descent backpropagation

net.performFcn = 'mse'; %Mean Squared Error

net.layers{1}.transferFcn = activation_func;
%tansig logsig purelin hardlim

net.layers{2}.transferFcn = 'purelin'; %output neuron is linear

% net.divideParam.trainRatio = 0.60; % training set [%]
% net.divideParam.valRatio = 0.20; % validation set [%]
% net.divideParam.testRatio = 0.20; % test set [%]

net.divideFcn= 'divideblock'; % divide the data manually
net.divideParam.trainRatio = 0.60; % training data
net.divideParam.valRatio= 0.20; % validation data 
net.divideParam.testRatio=  0.20;  % testing data


net.trainParam.epochs = 5000; %default epoc number
net.trainParam.show=1; %show after every iteration
net.trainParam.max_fail = 10; %Maximum validation failures

%train the network
[net,tr] = train(net,input_values,target_values);

%plot of functions
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
  'plotregression', 'plotfit'};

%test the network and network response after training
final_output = net(input_values); 
%e = gsubtract(target_values,final_output);
MSE = perform(net, target_values, final_output);

%simulating the outputs 
alloutputs = sim(net,input_values);

% Training, Validation and Test Performance
 trainTargets = target_values .* tr.trainMask{1};
 valTargets = target_values  .* tr.valMask{1};
 testTargets = target_values  .* tr.testMask{1};
% 
% MSE OUTPUTS of Training, Validation and Test Performance
 trainP = perform(net,trainTargets,final_output)
 valP = perform(net,valTargets,final_output)
 testP = perform(net,testTargets,final_output)

%Plots

% figure, plotperform(tr);
% figure, plottrainstate(tr);
% figure, plotfit(net, input_values, target_values);
% figure, plotregression(target_values, final_output);
% figure, ploterrhist(e);
end
%---------------------------------