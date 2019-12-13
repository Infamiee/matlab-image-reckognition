
function [accuracy,classifier,net] = train_model(categories)
%% stworzenie wczesniej nauczonej sieci
net = resnet50()
image_size = net.Layers(1).InputSize;
%% pobranie danych do trenowania modelu i podzielenie
[training_set,test_set,augmented_training_set,augmented_test_set] = learning_data(categories,image_size)
%% uczenie sieci nowymi obrazami
training_options = activations(net,augmented_training_set,'fc1000','MiniBatchSize',32,'OutputAs','columns');
training_labels = training_set.Labels;
classifier = fitcecoc(training_options,training_labels,'Learner','Linear','Coding','onevsall','ObservationsIn','columns');

%% testowanie sieci
test_options = activations(net,augmented_test_set,'fc1000','MiniBatchSize',32,'OutputAs','columns');
predict_labels = predict(classifier,test_options,'ObservationsIn','columns');
test_labels=test_set.Labels;

%% obliczenie dokladnosci uczenia
conf_mat = confusionmat(test_labels,predict_labels);
conf_mat = bsxfun(@rdivide, conf_mat,sum(conf_mat,2));
disp(conf_mat)
accuracy = mean(diag(conf_mat))

end