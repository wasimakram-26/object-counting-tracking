%% REAL-TIME OBJECT COUNTING AND TRACKING

clc; clear; close all;

%% Step 1: Video input
% Option 1: Webcam
% cam = webcam;
% Option 2: Video file
vid = VideoReader('objects_video.mp4');

%% Step 2: Background subtraction (for static background videos)
foregroundDetector = vision.ForegroundDetector('NumGaussians',3,'NumTrainingFrames',50);

%% Step 3: Blob analysis
blobAnalyzer = vision.BlobAnalysis('BoundingBoxOutputPort',true,'AreaOutputPort',true,'MinimumBlobArea',150);

%% Step 4: Process video frames
while hasFrame(vid)
    frame = readFrame(vid);
    
    % Convert to grayscale
    gray = rgb2gray(frame);
    
    % Detect foreground objects
    fgMask = step(foregroundDetector,gray);
    fgMask = imopen(fgMask, strel('rectangle',[3 3])); % clean mask
    fgMask = imclose(fgMask, strel('rectangle',[5 5]));
    fgMask = imfill(fgMask,'holes');
    
    % Analyze blobs
    [areas, boxes] = step(blobAnalyzer, fgMask);
    numObjects = numel(areas);
    
    % Display results
    outFrame = insertShape(frame,'Rectangle',boxes,'Color','red','LineWidth',2);
    outFrame = insertText(outFrame,[10 10],['Objects Counted: ', num2str(numObjects)],'FontSize',16,'BoxColor','yellow','BoxOpacity',0.6);
    
    imshow(outFrame);
    drawnow;
end