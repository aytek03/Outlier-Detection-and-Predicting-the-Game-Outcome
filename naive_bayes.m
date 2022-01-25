function [naive_con] = naive_bayes()

% AYTEKÝN YILDIZHAN
% N18147923
% CMP712 Machine Learning Project
%
% Naive Bayes classification algorithm is presented.


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


% bys = NaiveBayes.fit(new_data,winClass);
% Tahmin = bys.predict(new_data);
% Mat1 = confusionmat(winClass, Tahmin);

rng(1); % For reproducibility

%OUR DATA COMES FROM NCA_C with 2 features
classNames = {'A','B','C','D'}; % Class order

Mdl = fitcnb(our_data4,winClass,'ClassNames',classNames);

CVMdl = crossval(Mdl,'kfold',5);

genError = kfoldLoss(CVMdl)

% Tahmin = bys.predict(our_data4);
% naive_con = confusionmat(winClass, Tahmin);
% 

end