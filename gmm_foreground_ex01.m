%% Detecting Motions Using Gaussian Mixture MOdel
%
%
%
%https://ch.mathworks.com/help/vision/examples/detecting-cars-using-gaussian-mixture-models.html
%

%%


clear all; close all;
clc;

%%

FRAME_WIDTH = 1280;
FRAME_HEIGHT = 720;



vidDevice = imaq.VideoDevice('winvideo', 2, strcat('MJPG_',num2str(FRAME_WIDTH),'x',num2str(FRAME_HEIGHT)), ... % Acquire input video stream
 'ROI', [1 1 FRAME_WIDTH FRAME_HEIGHT], ...
 'ReturnedColorSpace', 'rgb');


foregroundDetector = vision.ForegroundDetector('NumGaussians', 3, ...
    'NumTrainingFrames', 50);


blobAnalysis = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', false, 'CentroidOutputPort', false, ...
    'MinimumBlobArea', 3500);

se = strel('square', 5); % morphological filter for noise removal

%% Train Foreground Detector
disp('Training');
for i = 1:150
    disp(strcat('step#',num2str(i)));
    frame = step(vidDevice); % read the next video frame
    foreground = step(foregroundDetector, frame);
end
disp('Training Complete');

%%

for i=1:400
   rgbFrame = step(vidDevice);
   rgbFrame=flip(rgbFrame,2);
   grayFrame = rgb2gray(rgbFrame);
   grayFrame = imgaussfilt(grayFrame,4);
   
   
   % Detect the foreground in the current video frame
    foreground = step(foregroundDetector, rgbFrame);

    % Use morphological opening to remove noise in the foreground
    filteredForeground = imopen(foreground, se);

    % Detect the connected components with the specified minimum area, and
    % compute their bounding boxes
    bbox = step(blobAnalysis, filteredForeground);

    % Draw bounding boxes around the detected cars
    result = insertShape(rgbFrame, 'Rectangle', bbox, 'Color', 'green','LineWidth', 5);

    % Display the number of cars found in the video frame
    numCars = size(bbox, 1);
    result = insertText(result, [10 10], numCars, 'BoxOpacity', 1, ...
        'FontSize', 14);
    figure(5),imshow(result);
%    step(videoPlayer, result);  % display the results
    
    
end
%%
release(vidDevice)
