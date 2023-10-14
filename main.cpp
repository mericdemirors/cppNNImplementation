#include "NN.h"

int main()
{
	pair<pair<float***, float**>, pair<int, int>> X_and_Y_AND_l_and_c_train = read_csv("housing_for_Cpp_train.csv");
	float*** Xs_train = (X_and_Y_AND_l_and_c_train.first).first;
	float** Ys_train = (X_and_Y_AND_l_and_c_train.first).second;
	int row_number_train = (X_and_Y_AND_l_and_c_train.second).first;
	int feature_number_train = (X_and_Y_AND_l_and_c_train.second).second-1;

	pair<pair<float***, float**>, pair<int, int>> X_and_Y_AND_l_and_c_valid = read_csv("housing_for_Cpp_valid.csv");
	float*** Xs_valid = (X_and_Y_AND_l_and_c_valid.first).first;
	float** Ys_valid = (X_and_Y_AND_l_and_c_valid.first).second;
	int row_number_valid = (X_and_Y_AND_l_and_c_valid.second).first;
	int feature_number_valid = (X_and_Y_AND_l_and_c_valid.second).second-1;
	
	pair<pair<float***, float**>, pair<int, int>> X_and_Y_AND_l_and_c_test = read_csv("housing_for_Cpp_test.csv");
	float*** Xs_test = (X_and_Y_AND_l_and_c_test.first).first;
	float** Ys_test = (X_and_Y_AND_l_and_c_test.first).second;
	int row_number_test = (X_and_Y_AND_l_and_c_test.second).first;
	int feature_number_test = (X_and_Y_AND_l_and_c_test.second).second-1;
	
	int epoch=100, batch_size=4, layer_number=3, layers[layer_number] = {feature_number_train,7,1};
	NN nn(layers, layer_number, "sigmoid", 0.05);
	
	float** losses = nn.fit(epoch, batch_size, Xs_train, Ys_train, row_number_train, Xs_valid, Ys_valid, row_number_valid, layers, layer_number);
	
	float** test_pred = nn.predict(Xs_test, row_number_test, layers, layer_number);
	float test_loss = nn.loss(Ys_test, test_pred, layers[layer_number-1], 1);
	cout << "test loss: " << test_loss << endl;
	
	plot(epoch, row_number_test, losses[0], losses[1], Ys_test, test_pred);
}
