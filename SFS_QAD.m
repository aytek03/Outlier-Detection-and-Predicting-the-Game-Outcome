function [] = SFS_QAD()

% AYTEKÝN YILDIZHAN
% N18147923
% CMP712 Machine Learning Project
%
%----------------------------------------------------------------
% This script is a sequential forward feature selection with
% Quadratic Discriminant Analysis

% our data and win_ratio are sent to ANN at the bottom of the code. 


% Our output are train, test and validation values.

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

% winClass2 = zeros(size(win_ratio_100));
% class_1a = 0;
% class_2a = 0;
% class_3a = 0;
% class_4a = 0;
% 
% for i=1:684
%  if (win_ratio_100(i) <= 35) 
%           winClass2(i,1) = 1; 
%           class_1a = class_1a + 1;
%  elseif (win_ratio_100(i) > 35 && win_ratio_100(i) <= 50)
%           winClass2(i,1) = 2;
%           class_2a = class_2a + 1;
%  elseif (win_ratio_100(i) > 50 && win_ratio_100(i) <= 65)
%           winClass2(i,1) = 3;
%           class_3a = class_3a + 1;
%  else 
%           winClass2(i,1) = 4;
%           class_4a = class_4a + 1;
%  end
% end


% c = cvpartition(winClass,'k',10);
% opts = statset('display','iter');
% fun = @(XT,yT,Xt,yt)...
%       (sum(~strcmp(yt,classify(Xt,XT,yT,'quadratic'))));
% 
% [fs,history] = sequentialfs(fun,our_data3,winClass,'cv',c,'options',opts);
           
%Starting Quadratic Discriminant Analysis

rng(8000,'twister');

holdoutCVP = cvpartition(winClass,'holdout',171)% test size 
egitimVerisi = our_data4(holdoutCVP.training,:);
egitimGrubu = winClass(holdoutCVP.training);


egitimVerisiG1 =egitimVerisi(grp2idx(egitimGrubu)==1,:);
egitimVerisiG2 = egitimVerisi(grp2idx(egitimGrubu)==2,:);
[h,p,ci,stat] = ttest2(egitimVerisiG1,egitimVerisiG2,[],[],'unequal');

[~,featureIdxSortbyP]= sort(p,2); % sort the features
testMCE =zeros(1,32);
resubMCE = zeros(1,32);
nfs = 1:1:32; %nfs = 5:5:70;
classf = @(xtrain,ytrain,xtest,ytest) ...
             sum(~strcmp(ytest,classify(xtest,xtrain,ytrain,'quadratic')));
resubCVP = cvpartition(length(winClass),'resubstitution')
for i=1:10 %6
   fs = featureIdxSortbyP(1:nfs(i));
   testMCE(i) = crossval(classf,our_data4(:,fs),winClass,'partition',holdoutCVP)...
       /holdoutCVP.TestSize;
   resubMCE(i) = crossval(classf,our_data4(:,fs),winClass,'partition',resubCVP)/...
       resubCVP.TestSize;
end  


corr(egitimVerisi(:,featureIdxSortbyP(1)),egitimVerisi(:,featureIdxSortbyP(2))) 
tenfoldCVP = cvpartition(egitimGrubu,'kfold',10);

fs1 = featureIdxSortbyP(1:32);

fsLocal = sequentialfs(classf,egitimVerisi(:,fs1),egitimGrubu,'cv',tenfoldCVP)
fs1(fsLocal)

testMCELocal = crossval(classf,our_data4(:,fs1(fsLocal)),winClass,'partition',...
    holdoutCVP)/holdoutCVP.TestSize


our_data5=our_data4(: , fs1(fsLocal));

our_data5 = our_data5';
win_ratio = win_ratio';
 
% our data and win_ratio are sent to ANN.

[MSE,alloutputs] = experiment1(2,'trainlm','tansig',our_data5,win_ratio);

%[MSE,alloutputs] = experiment1(64,'trainlm','tansig',our_data5,win_ratio);

%[MSE,alloutputs] = experiment1(128,'trainlm','tansig',our_data5,win_ratio);

end