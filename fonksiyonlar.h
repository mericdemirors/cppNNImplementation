#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std;
//------------------------------------------------------------------------------------------------------------------------//
float** create_matrix(int r, int c)
{
	float** m = (float**)malloc(r * sizeof(float*));

	for(int i=0; i<r; i++)
	{
		m[i] = (float*)malloc(c * sizeof(float));
		
		for(int j=0; j<c; j++)
			m[i][j] = 2*(static_cast <float> (rand()) / static_cast <float> (RAND_MAX)-0.5);
	}
	return m;
}
float** transpose(float** m, int r, int c)
{
	float** mt = create_matrix(c,r);
	for (int i=0; i<r; i++)
	{
		for (int j=0; j<c; j++)
			mt[j][i] = m[i][j];
	}		
	return mt;
}
float** copy_matrix(float** m, int r, int c)
{
	float** copy = create_matrix(r,c);
	for(int  i=0; i<r; i++)
		for(int j=0; j<c; j++)
			copy[i][j] = m[i][j];
	return copy;
}
void print_matrix(float** m, int r, int c)
{
	for (int i=0; i<r; i++)
	{
		for (int j=0; j<c; j++)
			cout << m[i][j] << ", ";
		
		cout << endl;
	}
	cout << endl;
}
//------------------------------------------------------------------------------------------------------------------------
float** elementwise_matrix_operations(float** m1, float** m2, int r, int c, string operation)
{
	float** m3 = create_matrix(r, c);

	for (int i=0; i<r; i++)
	{
		for (int j=0; j<c; j++)
		{
			if (operation == "+")
				m3[i][j] = m1[i][j] + m2[i][j];
			else if (operation == "-")
				m3[i][j] = m1[i][j] - m2[i][j];
			else if (operation == "*")
				m3[i][j] = m1[i][j] * m2[i][j];
			else if (operation == "/" and m2[i][j] != 0)
				m3[i][j] = m1[i][j] / m2[i][j];
			else
			{
				cout << "Unable to do given operation or division with 0.\n";
				exit(1);
			}
		}
	}
	return m3;
}
float** matrix_element_operations(float** m1, float element, int r, int c, string operation)
{
	float** m3 = create_matrix(r, c);
	
	for (int i=0; i<r; i++)
		for (int j=0; j<c; j++)
		{
			if (operation == "+")
				m3[i][j] = m1[i][j] + element;
			else if (operation == "-")
				m3[i][j] = m1[i][j] - element;
			else if (operation == "*")
				m3[i][j] = m1[i][j] * element;
			else if (operation == "/" and element != 0)
				m3[i][j] = m1[i][j] / element;
			else
			{
				cout << "Unable to do given operation or division with 0.\n";
				exit(1);
			}
		}

	return m3;
}
float** matrix_multipication(float** m1, float** m2, int r1, int c1, int r2, int c2)
{
	if (c1 != r2)
	{
		cout << "Unable to multiply martixs with this sizes.\n";
		exit(1);
	}
	
	float** m3 = create_matrix(r1, c2);


	for(int i=0; i<r1; i++)
		for (int j=0; j<c2; j++)
		{
			m3[i][j] = 0;
			for(int c1_r2=0; c1_r2<c1; c1_r2++)
				m3[i][j] = m3[i][j] + m1[i][c1_r2]*m2[c1_r2][j]; 
		}

	return m3;
}
//------------------------------------------------------------------------------------------------------------------------
float*** create_weights(int* layer_arr, int layer_number)
{
	float*** weights = (float***)malloc((layer_number-1) * sizeof(float**));
	
	for(int i=0; i<layer_number-1; i++)
		weights[i] = create_matrix(layer_arr[i],layer_arr[i+1]);

	return weights;
}
void print_weights(float*** weights, int* layer_arr, int layer_number)
{
	for (int i=0; i<layer_number-1; i++)
		print_matrix(weights[i], layer_arr[i], layer_arr[i+1]);
}
//------------------------------------------------------------------------------------------------------------------------
float*** create_biasses(int* layer_arr, int layer_number)
{
	float*** biasses = (float***)malloc((layer_number-1) * sizeof(float**));
	
	for(int i=0; i<layer_number-1; i++)
		biasses[i] = create_matrix(layer_arr[i+1],1);

	return biasses;
}

void print_biasses(float*** biasses, int* layer_arr, int layer_number)
{
	for (int i=0; i<layer_number-1; i++)
		print_matrix(biasses[i], layer_arr[i+1], 1);
}
//------------------------------------------------------------------------------------------------------------------------
// spesific functions for reading .csv and parsing rows to matrixs
float** line_to_matrix(string line, int c)
{
	line = line + ",";
	float** m = create_matrix(c,1);
	
	for(int i=0; i<c; i++)
	{
		string sub_line = line.substr(0, line.find(','));// line[0:till next ','];
		m[i][0] = stof(sub_line);
		line = line.substr(line.find(',') + 1, line.length() - line.find(','));// line[next ',' + 1: till end];
	}

	return m;
}

pair<pair<float***, float**>, pair<int, int>> read_csv(string file)
{
	int line_number = -1, column_number=1;
	string line="", last_line;
	ifstream counting_file(file);
	
	if(counting_file.is_open()) 
	{
		while(!counting_file.eof())
		{
			last_line=line;
			getline(counting_file, line);
			line_number = line_number + 1;
		}
		for(int i=0; last_line[i]!='\0'; i++)
		{
			if(last_line[i] == ',')
				column_number = column_number + 1;
		}
		counting_file.close();
	}
	

	pair<float***, float**> X_and_Y;
	pair<int, int> l_and_c;
	float*** Xarray = (float***)malloc(line_number * sizeof(float**));;
	float** Yarray = create_matrix(line_number, 1);

	ifstream mFile(file);
	int l=0;
	if(mFile.is_open()) 
	{
		while(!mFile.eof())
		{
			getline(mFile, line);
			if (line[0] == '\0')
				break;
				
			float** m = line_to_matrix(line, column_number);
			
			Xarray[l] = create_matrix(column_number-1, 1);
			for(int i=0; i<column_number-1; i++)
				Xarray[l][i][0] = m[i][0];
			
			Yarray[l][0] = m[column_number-1][0];
			
			l = l + 1;
		}

		mFile.close();
	}
	

	X_and_Y.first = Xarray;
	X_and_Y.second = Yarray;
	l_and_c.first = l;
	l_and_c.second = column_number;
	
	pair<pair<float***, float**>, pair<int, int>> to_return;
	to_return.first = X_and_Y;
	to_return.second = l_and_c;
		
	return to_return;
}
//------------------------------------------------------------------------------------------------------------------------
// plotting with python
void plot(int epoch, int test_row_number, float* train_losses, float* valid_losses, float** y_test, float** test_pred)
{
	string train_losses_arg = "", valid_losses_arg = "", y_test_arg = "", test_pred_arg = "";

	for(int i=0; i<epoch; i++)
		if (i!=epoch-1)
			train_losses_arg = train_losses_arg + to_string(train_losses[i]) + ",";
		else
			train_losses_arg = train_losses_arg + to_string(train_losses[i]);
	
		
	for(int i=0; i<epoch; i++)
		if (i!=epoch-1)
			valid_losses_arg = valid_losses_arg + to_string(valid_losses[i]) + ",";
		else
			valid_losses_arg = valid_losses_arg + to_string(valid_losses[i]);
		
	
	for(int i=0; i<test_row_number; i++)
		if(i!=test_row_number-1)
			y_test_arg = y_test_arg + to_string(y_test[0][i]) + ",";
		else
			y_test_arg = y_test_arg + to_string(y_test[0][i]);
	

	for(int i=0; i<test_row_number; i++)
		if(i!=test_row_number-1)
			test_pred_arg = test_pred_arg + to_string(test_pred[0][i]) + ",";
		else
			test_pred_arg = test_pred_arg + to_string(test_pred[0][i]);

	string command_str = "python3 plots.py "+train_losses_arg+" "+valid_losses_arg+" "+y_test_arg+" "+test_pred_arg;
	const char* command = command_str.c_str();

	system(command);
}





