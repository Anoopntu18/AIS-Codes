clc;
clear all;
close all;

digitDatasetPath = fullfile(matlabroot, 'toolbox', 'nnet', 'nndemos', 'nndatasets', 'Discriminating shape of objects');
data = imageDatastore(digitDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

[trainData] = splitEachLabel(data, 0.7, 'randomize');
count = trainData.countEachLabel;

layers = [
    imageInputLayer([1600 1000 3])
    convolution2dLayer(1, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(1, 'Stride', 2) 
    convolution2dLayer(2, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(1, 'Stride', 2)
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(1, 'Stride', 2)
    convolution2dLayer(4, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(1, 'Stride', 2)
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer
];

opt = trainingOptions('adam', 'MaxEpochs', 30, 'InitialLearnRate', 0.0001);
trainednet = trainNetwork(trainData, layers, opt);

[testData] = splitEachLabel(data, 7);
allclass = [];

for ii = 1:length(testData.Labels)
    I = readimage(testData, ii);
    class = classify(trainednet, I);
    allclass = [allclass class];
    figure(2),
    subplot(7, 7, ii)
    imshow(I)
    title(char(class))
end

Predicted = allclass;
figure, m = plotconfusion(testData.Labels, Predicted');
