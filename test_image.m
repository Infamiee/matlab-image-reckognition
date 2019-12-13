function label = test_image(filepath,net,classifier)
img = imread(filepath);
image_size = net.Layers(1).InputSize;

augmented_img = augmentedImageDatastore(image_size,img,'ColorPreprocessing','gray2rgb');
img_options = activations(net,augmented_img,'fc1000','MiniBatchSize',32,'OutputAs','columns');

label = predict(classifier,img_options,'ObservationsIn','columns')
c=strrep(cellstr(label),'_',' ')
imshow(img)
title(["Kategoria",c])
end