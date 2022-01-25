function [NCA] = NCA_Class()

% AYTEKÝN YILDIZHAN
% N18147923
% CMP712 Machine Learning Project
%
%----------------------------------------------------------------
% Feature selection using neighborhood component 
% analysis for classification

clc
clear all

A = importdata('team_season_1979.txt');
 
our_data = A.data;
our_data2 = A.data; %without year

B=A.textdata;
B1 = B;
B1(1,:) = [];
B2 = B1(:,2);
B3 = cell2mat(B2);
B4 = str2num(B3); %we take 'year' part.

%we put 'year' part at the beginning of our data
our_data = [B4 A.data];

%calculating win ratio
for i=1:684

    win_ratio (i,1) = our_data(i,end-1)/(our_data(i,end-1)+our_data(i,end));

end

%we put 'year' part at the beginning of our data
our_data3 = our_data;

our_data(:,end) = []; 
our_data(:,end) = [];

our_data4 = our_data; % no win and lost.
 
%our_data = [our_data win_ratio]; %no win ratio added.

win_ratio_100 = 100 * win_ratio;

%winClass = zeros(size(win_ratio_100));
class_1 = 0;
class_2 = 0;
class_3 = 0;
class_4 = 0;

for i=1:684
 if (win_ratio_100(i) <= 35) 
    winClass{i,1} = 'A'; 
    class_1 = class_1 + 1;
    elseif (win_ratio_100(i) > 35 && win_ratio_100(i) <= 50)
    winClass{i,1} = 'B';
  class_2 = class_2 + 1;
      elseif (win_ratio_100(i) > 50 && win_ratio_100(i) <= 65)
    winClass{i,1} = 'C';
  class_3 = class_3 + 1;
    else 
    winClass{i,1} = 'D';
  class_4 = class_4 + 1;
 end
end

%NCA STARTING HERE.....

rng(1); % For reproducibility
cvp = cvpartition(winClass,'holdout',171);

Xtrain = our_data4(cvp.training,:);
ytrain = winClass(cvp.training,:);
Xtest  = our_data4(cvp.test,:);
ytest  = winClass(cvp.test,:);


nca = fscnca(Xtrain,ytrain,'FitMethod','none');
L = loss(nca,Xtest,ytest);

nca = fscnca(Xtrain,ytrain,'FitMethod','exact','Lambda',0,...
      'Solver','sgd','Standardize',true);
L = loss(nca,Xtest,ytest);

cvp = cvpartition(ytrain,'kfold',5);
numvalidsets = cvp.NumTestSets;


n = length(ytrain);
lambdavals = linspace(0,20,20)/n;
lossvals = zeros(length(lambdavals),numvalidsets);

%STARTING FILTER HERE....

for i = 1:length(lambdavals)
    for k = 1:numvalidsets
        X = Xtrain(cvp.training(k),:);
        y = ytrain(cvp.training(k),:);
        Xvalid = Xtrain(cvp.test(k),:);
        yvalid = ytrain(cvp.test(k),:);

        nca = fscnca(X,y,'FitMethod','exact', ...
             'Solver','sgd','Lambda',lambdavals(i), ...
             'IterationLimit',30,'GradientTolerance',1e-4, ...
             'Standardize',true);
                  
        lossvals(i,k) = loss(nca,Xvalid,yvalid,'LossFunction','classiferror');
    end
end


meanloss = mean(lossvals,2);

figure()
plot(lambdavals,meanloss,'ro-')
xlabel('Lambda')
ylabel('Loss (MSE)')
grid on

[~,idx] = min(meanloss) % Find the index
bestlambda = lambdavals(idx) % Find the best lambda value
bestloss = meanloss(idx)


nca = fscnca(Xtrain,ytrain,'FitMethod','exact','Solver','sgd',...
    'Lambda',bestlambda,'Standardize',true,'Verbose',1);


tol    = 0.02;
selidx = find(nca.FeatureWeights > tol*max(1,max(nca.FeatureWeights)))


L = loss(nca,Xtest,ytest)

features = Xtrain(:,selidx);

selidx = selidx'; %OUR SELECTED FEATURE

our_data5=our_data4(: , selidx);

our_data5 = our_data5';
win_ratio = win_ratio';
 
[MSE,alloutputs] = experiment1(2,'trainlm','tansig',our_data5,win_ratio);

%[MSE,alloutputs] = experiment1(64,'trainlm','tansig',our_data5,win_ratio);

%[MSE,alloutputs] = experiment1(128,'trainlm','tansig',our_data5,win_ratio);

% without filter
% NCA = fscnca(our_data4,winClass); %classificaiton
% 
% figure()
% plot(NCA.FeatureWeights,'ro') %nca filter
% grid on
% xlabel('Feature index')
% ylabel('Feature weight');

end