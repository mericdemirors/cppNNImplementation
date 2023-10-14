#include "fonksiyonlar.h"

#include <stdlib.h>
#include<algorithm>
#include <math.h>

using namespace std;


class NN
{
private:
	// activation and derivative_activation functions
	float af(float f){return (this->*activ_func)(f);}	
	float (NN::*activ_func)(float f);
	float RelU(float f) { return max(0.0f, f); }
	float sigmoid(float f) {return (exp(f) / (exp(f)+1));}
	float tan(float f) {return (exp(f) - exp(-1*f)) / (exp(f) + exp(-1*f));}
	
	float daf(float f){return (this->*deriv_activ_func)(f);}
	float (NN::*deriv_activ_func)(float f);
	float derivative_RelU(float f) {if (f>=0) return 1; else return 0;}
	float derivative_sigmoid(float f) {return sigmoid(f)*(1-sigmoid(f));}
	float derivative_tan(float f) {return 1 - tanh(f)*tanh(f);}

public:
	int alpha;
	float*** weights, ***biasses;
//.......................................................................................................................//
	NN(int* layers, int layer_number, string activate_function, float learning_rate)
	{
		alpha = learning_rate;
		
		if (activate_function == "RelU")
		{
			activ_func = &NN::RelU;
			deriv_activ_func = &NN::derivative_RelU;		
		}
		else if (activate_function == "sigmoid")
		{
			activ_func = &NN::sigmoid;
			deriv_activ_func = &NN::derivative_sigmoid;		
		}
		else if (activate_function == "tan")
		{
			activ_func = &NN::tan;
			deriv_activ_func = &NN::derivative_tan;		
		}
		
		weights = create_weights(layers, layer_number);
		biasses = create_biasses(layers, layer_number);
	}
//.......................................................................................................................//	
	void print(int* layers, int layer_number)
	{
		cout << "WEIGHTS:\n\n";
		print_weights(weights, layers, layer_number);
		cout << "\n\nBIASSES:\n\n";
		print_biasses(biasses, layers, layer_number);
	}
//.......................................................................................................................//
// MSE loss function and its derivative
	float loss(float** actual, float** pred, int r, int c)
	{
		float** diff = elementwise_matrix_operations(actual, pred, r, c, "-");
		float** diff_square = elementwise_matrix_operations(diff, diff, r, c, "*");

		float sum=0;
		for(int i=0; i<r; i++)
			for(int j=0; j<c; j++)
				sum = sum + diff_square[i][j];

		free(diff);
		free(diff_square);		
		return sum/(r*c);
	}
	float derivative_loss(float** actual, float** pred, int r, int c)
	{
		float** diff = elementwise_matrix_operations(actual, pred, r, c, "-");

		float sum=0;
		for(int i=0; i<r; i++)
			for(int j=0; j<c; j++)
				sum = sum + diff[i][j];

		free(diff);
		return (2/(r*c)) * sum;
	}
//.......................................................................................................................//
// passing martrixs to activation and derivative_activation functions
	float** activate(float** m, int r, int c)
	{
		float** active_m = create_matrix(r,c);
		for(int i=0; i<r; i++)
			for(int j=0; j<r; j++)
				active_m[i][j] = af(m[i][j]);
		return active_m;
	}	
	float** derivative_activate(float** m, int r, int c)
	{
		float** derivctive_m = create_matrix(r,c);
		for(int i=0; i<r; i++)
			for(int j=0; j<r; j++)
				derivctive_m[i][j] = daf(m[i][j]);
		return derivctive_m;
	}	
//.......................................................................................................................//	
	float**** forward_prop(float** X, int* layers, int layer_number)
	{
		// returning Zs_and_As: [0]->Zs, [1]->As
		// X is layers[0]x1 type input matrix
		float** WT = NULL;
		float** Xcopy = copy_matrix(X, layers[0], 1);
		float**** Zs_and_As = (float****)malloc(2 * sizeof(float***));
		Zs_and_As[0] = (float***)malloc((layer_number-1) * sizeof(float**));
		Zs_and_As[1] = (float***)malloc((layer_number-1) * sizeof(float**));

		for(int i=0; i<layer_number-1; i++)
		{
			float** W = weights[i];
			float** B = biasses[i];

			WT = transpose(W, layers[i], layers[i+1]);

			float** WT_mult_X = matrix_multipication(WT, Xcopy, layers[i+1], layers[i], layers[i], 1);			
			Zs_and_As[0][i] = elementwise_matrix_operations(WT_mult_X, B, layers[i+1], 1, "+");

			Xcopy = activate(Zs_and_As[0][i], layers[i+1], 1);
			Zs_and_As[1][i] = Xcopy;
			
			free(WT_mult_X);
		}	
				
		return Zs_and_As;
	}
//.......................................................................................................................//	
	float**** back_prop(float** X, float* Y, float*** Zs, float*** As, int* layers, int layer_number)
	{
		// returning We_and_Be: [0]->We, [1]->Be
		// X is layers[0]x1 type input matrix, and Y is layers[-1]x1 type matrix.

		float** Xcopy = copy_matrix(X, layers[0], 1);
		float** Ymatrix = create_matrix(layers[layer_number-1], 1);
		for(int i=0; i<layers[layer_number-1]; i++)
			Ymatrix[i][0] = Y[i];

		float**** We_and_Be = (float****)malloc(2 * sizeof(float***));
		We_and_Be[0] = (float***)malloc((layer_number-1) * sizeof(float**));
		We_and_Be[1] = (float***)malloc((layer_number-1) * sizeof(float**));

		float ** error = NULL;
		for(int i=layer_number-2; i>=0; i--)
		{
			// calculating (bias) error
			if (i==layer_number-2) //output layer
			{
				float pred_loss = derivative_loss(Ymatrix, As[i] ,layers[layer_number-1], 1);
				float** deriv_activ = derivative_activate(Zs[i], layers[layer_number-1], 1);

				deriv_activ = matrix_element_operations(deriv_activ, pred_loss, layers[layer_number-1], 1, "*");
				error = deriv_activ;
				We_and_Be[1][layer_number-2] = error;
			}
			else
			{
				float** deriv_activ = derivative_activate(Zs[i], layers[i+1], 1);
				float** W_mult_e = matrix_multipication(weights[i+1], error, layers[i+1], layers[i+2], layers[i+2], 1);
				error = elementwise_matrix_operations(W_mult_e, deriv_activ, layers[i+1], 1, "*");
				We_and_Be[1][i] = error;
				free(W_mult_e);
				free(deriv_activ);
			}

			// calculating weight error
			if (i == 0) // input
			{
				float** errorT = transpose(error, layers[1], 1);
				We_and_Be[0][0] = matrix_multipication(X, errorT, layers[0], 1, 1, layers[1]);
				free(errorT);
			}
			else
			{
				float** errorT = transpose(error, layers[i+1], 1);
				We_and_Be[0][i] = matrix_multipication(As[i-1], errorT, layers[i] , 1 , 1, layers[i+1]);
				free(errorT);
			}				
		}
		
		return We_and_Be;
	}
//.......................................................................................................................//		
	float** predict(float*** Xs, int X_number, int* layers, int layer_number)
	{
		float** predictions = create_matrix(layers[layer_number-1], 1);
		for(int i=0; i<X_number; i++)
		{
			float** X = Xs[i];
			float**** Zs_and_As = forward_prop(X, layers, layer_number);
			predictions[i] = Zs_and_As[1][layer_number-2][0];
		}
		return predictions;
	}
//.......................................................................................................................//
	void update_parameters(float*** We, float*** Be, int* layers, int layer_number)
	{
		for(int i=0; i<layer_number-1; i++)
		{
			We[i] = matrix_element_operations(We[i], alpha, layers[i], layers[i+1], "*");
			Be[i] = matrix_element_operations(Be[i], alpha, layers[i+1], 1, "*");

			weights[i] = elementwise_matrix_operations(weights[i], We[i], layers[i], layers[i+1], "-");
			biasses[i] = elementwise_matrix_operations(biasses[i], Be[i], layers[i+1], 1, "-");
		}
	}
//.......................................................................................................................//
	float**** gradient_descent(float*** Xs, float** Ys, int* layers, int layer_number, int row_number)
	{
		float**** total = (float****)malloc(2 * sizeof(float***));
		total[0] = (float***)malloc((layer_number-1) * sizeof(float**));
		total[1] = (float***)malloc((layer_number-1) * sizeof(float**));

		for(int i=0; i<row_number; i++)
		{
			float**** Zs_and_As = forward_prop(Xs[i], layers, layer_number);
			float**** We_and_Be = back_prop(Xs[i], Ys[i], Zs_and_As[0], Zs_and_As[1], layers, layer_number);
			if(i==0)
			{
				for(int j=0; j<layer_number-1; j++)
				{
					total[0][j] = copy_matrix(We_and_Be[0][j], layers[j], layers[j+1]);
					total[1][j] = copy_matrix(We_and_Be[1][j], layers[j+1], 1);
				}
			}
			else
			{
				for(int j=0; j<layer_number-1; j++)
				{
					total[0][j] = elementwise_matrix_operations(total[0][j], We_and_Be[0][j], layers[j], layers[j+1], "+");
					total[1][j] = elementwise_matrix_operations(total[1][j], We_and_Be[1][j], layers[j+1], 1, "+");
				}
			} 
		}
		for(int j=0; j<layer_number-1; j++)
		{
			total[0][j] = matrix_element_operations(total[0][j], row_number, layers[j], layers[j+1], "/");
			total[1][j] = matrix_element_operations(total[1][j], row_number, layers[j+1], 1, "/");
		}
			
		return total;
	}

//.......................................................................................................................//
	float** fit(int epoch, int batch_size, float*** Xs, float** Ys, int row_number, float*** valid_Xs, float** valid_Ys, int valid_row_number, int* layers, int layer_number, bool verbose=true)
	{
		float** losses = create_matrix(2,epoch); // losses[0]:train losses, losses[1]:validation losses, 
		float current_learning_rate = alpha;
		for(int e=0; e<epoch; e++)
		{
			if (e!=0 && e%100==0)
				current_learning_rate = current_learning_rate * 0.25;
			if (verbose && e!=0 && e%(int)sqrt(epoch)==0)
				cout << "epoch:" << e+1 << " ----- ";
		
		
			// shuffling Xs and Ys
			for(int i=0; i<row_number; i++)
			{
				int swap_ind = (rand() % row_number);

				float** tempX = Xs[i];
				Xs[i] = Xs[swap_ind];
				Xs[swap_ind] = tempX;			
				
				float* tempY = Ys[i];
				Ys[i] = Ys[swap_ind];
				Ys[swap_ind] = tempY;
			}


			// split Xs and Ys to batches
			int batch_number = row_number / batch_size, reduntant_batch_size = batch_size;
			if (row_number % batch_size !=0 )
			{
				batch_number = batch_number + 1;
				reduntant_batch_size = row_number % batch_size;
			}
			float**** all_X_batches = (float****)malloc(batch_number * sizeof(float***));
			float*** all_Y_batches = (float***)malloc(batch_number * sizeof(float**));

			for(int i=0; i<batch_number; i++)
			{
				if (i!= batch_number-1)
				{
					all_X_batches[i] = (float***)malloc(batch_size * sizeof(float**));
					all_Y_batches[i] = (float**)malloc(batch_size * sizeof(float*));
					
					for(int j=batch_size*i; j<batch_size*(i+1); j++)
					{
						all_X_batches[i][j%batch_size] = Xs[j];
						all_Y_batches[i][j%batch_size] = Ys[j];
					}
				}
				else
				{
					all_X_batches[i] = (float***)malloc(reduntant_batch_size * sizeof(float**));
					all_Y_batches[i] = (float**)malloc(reduntant_batch_size * sizeof(float*));
					
					for(int j=batch_size*i; j<row_number; j++)
					{
						all_X_batches[i][j%batch_size] = Xs[j];
						all_Y_batches[i][j%batch_size] = Ys[j];
					}
				}
			}
			
			
			// for each batch:
			//     calculate gradient_descent total error and update parameters
			for(int i=0; i<batch_number; i++)
			{
				if (i!= batch_number-1)
				{
					float**** total_We_and_Be = gradient_descent(all_X_batches[i], all_Y_batches[i], layers, layer_number, batch_size);
					update_parameters(total_We_and_Be[0], total_We_and_Be[1], layers, layer_number);
				}
				else
				{
					float**** total_We_and_Be = gradient_descent(all_X_batches[i], all_Y_batches[i], layers, layer_number, reduntant_batch_size);
					update_parameters(total_We_and_Be[0], total_We_and_Be[1], layers, layer_number);	
				}
			}
		

			// calculate train and validation loss
			// losses[0][e] = train_loss, losses[1][e] = valid_loss
			float** train_pred = predict(Xs, row_number, layers, layer_number);
			float train_loss = loss(Ys, train_pred, layers[layer_number-1], 1);
			float** valid_pred = predict(valid_Xs, valid_row_number, layers, layer_number);
			float valid_loss = loss(valid_Ys, valid_pred, layers[layer_number-1], 1);
			losses[0][e] = train_loss;
			losses[1][e] = valid_loss;


			if (verbose && e!=0 && e%(int)sqrt(epoch)==0)
				cout << "train loss:" << train_loss << "-----" << "validation loss:" << valid_loss << "\n";
		}
		
		return losses;
	}
};
