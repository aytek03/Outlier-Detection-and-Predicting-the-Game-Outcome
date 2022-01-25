function [] = No_SFS_ANN()

% AYTEKÝN YILDIZHAN
% N18147923
% CMP712 Machine Learning Project
%
% ANN model with two hidden layer.
% It uses no feature selection algorithm.

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

our_data3 = our_data;

our_data(:,end) = []; 
our_data(:,end) = [];

our_data4 = our_data; % no win and lost.
 
%our_data = [our_data win_ratio]; %no win ratio added.

win_ratio_100 = 100 * win_ratio;

% winClass = zeros(size(win_ratio_100));
class_1 = 0;
class_2 = 0;
class_3 = 0;
class_4 = 0;

for i=1:684
 if (win_ratio_100(i) <= 35) 
          winClass(i,1) = 1; 
          class_1 = class_1 + 1;
 elseif (win_ratio_100(i) > 35 && win_ratio_100(i) <= 50)
          winClass(i,1) = 2;
          class_2 = class_2 + 1;
 elseif (win_ratio_100(i) > 50 && win_ratio_100(i) <= 65)
          winClass(i,1) = 3;
          class_3 = class_3 + 1;
 else 
          winClass(i,1) = 4;
          class_4 = class_4 + 1;
 end
end

our_data4 = our_data4';


winClass = [winClass winClass winClass winClass];

winClass = winClass';

winClass2 = winClass;

%CLASS DEFINITION
for i=1:684
if (winClass2(1,i) == 1)
    winClass2(2,i)=0;
    winClass2(3,i)=0;
    winClass2(4,i)=0;

elseif (winClass2(2,i) == 2)
    winClass2(2,i) = 1;
    winClass2(1,i)=0;
    winClass2(3,i)=0;
    winClass2(4,i)=0;
    
elseif (winClass2(3,i) == 3)
    winClass2(1,i)=0;
    winClass2(3,i) = 1;
    winClass2(2,i)=0;
    winClass2(4,i)=0;
    
elseif (winClass2(4,i) == 4)
    winClass2(4,i) = 1;
    winClass2(1,i)=0;
    winClass2(2,i)=0;
    winClass2(3,i)=0;

end
end
    

[MSE,alloutputs] = experiment2(2,2,'trainlm','tansig','tansig',our_data4,winClass2);
%[MSE,alloutputs] = experiment2(2,2,'trainlm','logsig','logsig',our_data4,winClass2);
%[MSE,alloutputs] = experiment2(2,2,'trainlm','purelin','purelin',our_data4,winClass2);
%[MSE,alloutputs] = experiment2(2,2,'trainlm','hardlim','hardlim',our_data4,winClass2);

%[MSE,alloutputs] = experiment2(64,64,'trainlm','tansig','tansig',our_data4,winClass2);
%[MSE,alloutputs] = experiment2(64,64,'trainlm','logsig','logsig',our_data4,winClass2);
%[MSE,alloutputs] = experiment2(64,64,'trainlm','purelin','purelin',our_data4,winClass2);
%[MSE,alloutputs] = experiment2(64,64,'trainlm','hardlim','hardlim',our_data4,winClass2);


%[MSE,alloutputs] = experiment2(128,128,'trainlm','tansig','tansig',our_data4,winClass2);
%[MSE,alloutputs] = experiment2(128,128,'trainlm','logsig','logsig',our_data4,winClass2);
%[MSE,alloutputs] = experiment2(128,128,'trainlm','purelin','purelin',our_data4,winClass2);
%[MSE,alloutputs] = experiment2(128,128,'trainlm','hardlim','hardlim',our_data4,winClass2);

end