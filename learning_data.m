function [training_set,test_set,augmented_training_set,augmented_test_set] = learning_data(categories,image_size)
%% wstepne przetwarzanie zdj?? do rozdzielczo?ci u?ywanej przez sie? konwolucyjn? resnet50 i rozdzielenie na zestaw treningowy i testowy
% 80% danych treningowych i 20% testowych
root_folder = fullfile('Categories');
imds = imageDatastore(fullfile(root_folder,categories),'LabelSource','foldernames');
tbl = countEachLabel(imds);
min_set_count = min(tbl{:,2});
imds = splitEachLabel(imds,min_set_count,'randomize');
[training_set,test_set] = splitEachLabel(imds,0.2,'randomize');

augmented_training_set = augmentedImageDatastore(image_size,training_set,'ColorPreprocessing','gray2rgb');
augmented_test_set = augmentedImageDatastore(image_size,test_set,'ColorPreprocessing','gray2rgb');
end