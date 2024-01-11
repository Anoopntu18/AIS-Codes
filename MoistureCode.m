clc; clear all;close all;
syms m
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
'nndatasets','Mositure Range Modified Dataset');
data = imageDatastore(digitDatasetPath, ...
'IncludeSubfolders',true,'LabelSource','foldernames');

[trainData, testData] = splitEachLabel(data, 0.7, 'randomize');

layers = [
    imageInputLayer([1056 640 3])
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
    fullyConnectedLayer(9)
    softmaxLayer
    classificationLayer
];

opt = trainingOptions('adam', 'MaxEpochs', 10, 'InitialLearnRate', 0.0001, 'ExecutionEnvironment', 'auto');
trainednet = trainNetwork(trainData, layers, opt);

allclass = [];
for ii = 1:length(testData.Labels)
    I = readimage(testData, ii);
    class = classify(trainednet, I);
    allclass = [allclass class];
    figure(2),
    subplot(24, 24, ii)
    imshow(I)
    title(char(class))
end

Predicted = allclass;
figure, m = plotconfusion(testData.Labels, Predicted');
