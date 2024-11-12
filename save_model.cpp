#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include <fstream>
#include <unordered_map>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <algorithm>

void saveData(std::string fileName, Eigen::MatrixXf  matrix)
{
	//https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
	const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");

	std::ofstream file(fileName);
	if (file.is_open())
	{
		file << matrix.format(CSVFormat);
		file.close();
	}
}

Eigen::MatrixXf openData(std::string fileToOpen)
{

	// the inspiration for creating this function was drawn from here (I did NOT copy and paste the code)
	// https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix
	
	// the input is the file: "fileToOpen.csv":
	// a,b,c
	// d,e,f
	// This function converts input file data into the Eigen matrix format



	// the matrix entries are stored in this variable row-wise. For example if we have the matrix:
	// M=[a b c 
	//	  d e f]
	// the entries are stored as matrixEntries=[a,b,c,d,e,f], that is the variable "matrixEntries" is a row vector
	// later on, this vector is mapped into the Eigen matrix format
	std::vector<float> matrixEntries;

	// in this object we store the data from the matrix
	std::ifstream matrixDataFile(fileToOpen);

	// this variable is used to store the row of the matrix that contains commas 
	std::string matrixRowString;

	// this variable is used to store the matrix entry;
	std::string matrixEntry;

	// this variable is used to track the number of rows
	int matrixRowNumber = 0;


	while (getline(matrixDataFile, matrixRowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
	{
		std::stringstream matrixRowStringStream(matrixRowString); //convert matrixRowString that is a string to a stream variable.

		while (getline(matrixRowStringStream, matrixEntry, ',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
		{
			matrixEntries.push_back(stod(matrixEntry));   //here we convert the string to double and fill in the row vector storing all the matrix entries
		}
		matrixRowNumber++; //update the column numbers
	}

	// here we convet the vector variable into the matrix and return the resulting object, 
	// note that matrixEntries.data() is the pointer to the first memory location at which the entries of the vector matrixEntries are stored;
	return Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrixEntries.data(), matrixRowNumber, matrixEntries.size() / matrixRowNumber);

}

class ModelConverter {
  
public:

  torch::jit::script::Module nn_model;

  Eigen::VectorXf max_thrust;
  Eigen::VectorXf min_thrust;
  Eigen::VectorXf max_torque;
  Eigen::VectorXf min_torque;
  float max_u;
  float min_u;

  Eigen::VectorXf bias_control_net_layer_1;
  Eigen::MatrixXf weight_control_net_layer_1;

  Eigen::VectorXf bias_control_net_layer_2;
  Eigen::MatrixXf weight_control_net_layer_2;

  Eigen::VectorXf bias_control_net_layer_3;
  Eigen::MatrixXf weight_control_net_layer_3;

  Eigen::VectorXf bias_allocation_net_layer_1;
  Eigen::MatrixXf weight_allocation_net_layer_1;

  Eigen::VectorXf bias_allocation_net_layer_2;
  Eigen::MatrixXf weight_allocation_net_layer_2;

  ModelConverter(const std::string& filePath) {
    torch::jit::script::Module nn_model;
    torch::Device device(torch::kCPU);

    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      this->nn_model = torch::jit::load(filePath, device);

      for (const auto& p : this->nn_model.named_parameters()){
        auto par = p.value;
        if (p.name == "control_stack.0.bias"){
          this->bias_control_net_layer_1 = Eigen::Map<const Eigen::VectorXf>(par.data_ptr<float>(), par.size(0));
        }
        else if (p.name == "control_stack.0.weight"){
          this->weight_control_net_layer_1 = Eigen::Map<const Eigen::MatrixXf>(par.data_ptr<float>(), par.size(1), par.size(0)).transpose();
        }
        else if (p.name == "control_stack.2.bias"){
          this->bias_control_net_layer_2 = Eigen::Map<const Eigen::VectorXf>(par.data_ptr<float>(), par.size(0));
        }
        else if (p.name == "control_stack.2.weight"){
          this->weight_control_net_layer_2 = Eigen::Map<const Eigen::MatrixXf>(par.data_ptr<float>(), par.size(1), par.size(0)).transpose();
        }
        else if (p.name == "control_stack.4.bias"){
          this->bias_control_net_layer_3 = Eigen::Map<const Eigen::VectorXf>(par.data_ptr<float>(), par.size(0));
        }
        else if (p.name == "control_stack.4.weight"){
          this->weight_control_net_layer_3 = Eigen::Map<const Eigen::MatrixXf>(par.data_ptr<float>(), par.size(1), par.size(0)).transpose();
        }
        else if (p.name == "allocation_stack.0.bias"){
          this->bias_allocation_net_layer_1 = Eigen::Map<const Eigen::VectorXf>(par.data_ptr<float>(), par.size(0));
        }
        else if (p.name == "allocation_stack.0.weight"){
          this->weight_allocation_net_layer_1 = Eigen::Map<const Eigen::MatrixXf>(par.data_ptr<float>(), par.size(1), par.size(0)).transpose();
        }
        else if (p.name == "allocation_stack.2.bias"){
          this->bias_allocation_net_layer_2 = Eigen::Map<const Eigen::VectorXf>(par.data_ptr<float>(), par.size(0));
        }
        else if (p.name == "allocation_stack.2.weight"){
          this->weight_allocation_net_layer_2 = Eigen::Map<const Eigen::MatrixXf>(par.data_ptr<float>(), par.size(1), par.size(0)).transpose();
        }
      }

      std::cout << "Parameters loaded\n";

      for (auto p: this->nn_model.named_attributes()){
        auto val = p.value;
        if(p.name == "max_thrust"){
          torch::Tensor par = val.toTensor();
          this->max_thrust = Eigen::Map<const Eigen::VectorXf>(par.data_ptr<float>(), par.size(0));
        }
        else if (p.name == "min_thrust"){
          torch::Tensor par = val.toTensor();
          this->min_thrust = Eigen::Map<const Eigen::VectorXf>(par.data_ptr<float>(), par.size(0));
        }
        else if (p.name == "max_torque"){
          torch::Tensor par = val.toTensor();
          this->max_torque = Eigen::Map<const Eigen::VectorXf>(par.data_ptr<float>(), par.size(0));
        }
        else if (p.name == "min_torque"){
          torch::Tensor par = val.toTensor();
          this->min_torque = Eigen::Map<const Eigen::VectorXf>(par.data_ptr<float>(), par.size(0));
        }
        else if (p.name == "max_u"){
          torch::Tensor par = val.toTensor();
          this->max_u = float(par.data_ptr<float>()[0]);
        }
        else if (p.name == "min_u"){
          torch::Tensor par = val.toTensor();
          this->min_u = float(par.data_ptr<float>()[0]);
        }
      }

    }
    catch (const c10::Error& e) {
      std::cerr << "error loading the model\n";
    }

}
};



int main() {
    std::string filePath = "/home/welf/workspaces/training_hexbug/aerial_gym_dev/aerial_gym/rl_training/sample_factory/sim_to_real/deployment/deployed_models/quadBug.pt";
    ModelConverter converter(filePath);

    std::string path = "../model_files/";
    saveData(path + "bias_control_net_layer_1.csv", converter.bias_control_net_layer_1);
    saveData(path + "weight_control_net_layer_1.csv", converter.weight_control_net_layer_1);
    saveData(path + "bias_control_net_layer_2.csv", converter.bias_control_net_layer_2);
    saveData(path + "weight_control_net_layer_2.csv", converter.weight_control_net_layer_2);
    saveData(path + "bias_control_net_layer_3.csv", converter.bias_control_net_layer_3);
    saveData(path + "weight_control_net_layer_3.csv", converter.weight_control_net_layer_3);
    saveData(path + "bias_allocation_net_layer_1.csv", converter.bias_allocation_net_layer_1);
    saveData(path + "weight_allocation_net_layer_1.csv", converter.weight_allocation_net_layer_1);
    saveData(path + "bias_allocation_net_layer_2.csv", converter.bias_allocation_net_layer_2);
    saveData(path + "weight_allocation_net_layer_2.csv", converter.weight_allocation_net_layer_2);
    saveData(path + "max_thrust.csv", converter.max_thrust);
    saveData(path + "min_thrust.csv", converter.min_thrust);
    saveData(path + "max_torque.csv", converter.max_torque);
    saveData(path + "min_torque.csv", converter.min_torque);
    Eigen::Vector2f lim_u;
    lim_u << converter.min_u, converter.max_u;
    saveData(path + "lim_u.csv", lim_u);

    std::cout << "Eigen version: " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << std::endl;
    
    Eigen::VectorXf bias_control_net_layer_1 = openData(path + "bias_control_net_layer_1.csv");
    Eigen::MatrixXf weight_control_net_layer_1 = openData(path + "weight_control_net_layer_1.csv");
    Eigen::VectorXf bias_control_net_layer_2 = openData(path + "bias_control_net_layer_2.csv");
    Eigen::MatrixXf weight_control_net_layer_2 = openData(path + "weight_control_net_layer_2.csv");
    Eigen::VectorXf bias_control_net_layer_3 = openData(path + "bias_control_net_layer_3.csv");
    Eigen::MatrixXf weight_control_net_layer_3 = openData(path + "weight_control_net_layer_3.csv");
    Eigen::VectorXf bias_allocation_net_layer_1 = openData(path + "bias_allocation_net_layer_1.csv");
    Eigen::MatrixXf weight_allocation_net_layer_1 = openData(path + "weight_allocation_net_layer_1.csv");
    Eigen::VectorXf bias_allocation_net_layer_2 = openData(path + "bias_allocation_net_layer_2.csv");
    Eigen::MatrixXf weight_allocation_net_layer_2 = openData(path + "weight_allocation_net_layer_2.csv");
    Eigen::VectorXf max_thrust = openData(path + "max_thrust.csv");
    Eigen::VectorXf min_thrust = openData(path + "min_thrust.csv");
    Eigen::VectorXf max_torque = openData(path + "max_torque.csv");
    Eigen::VectorXf min_torque = openData(path + "min_torque.csv");
    Eigen::VectorXf limits_u = openData(path + "lim_u.csv");

    // forward path
    Eigen::VectorXf input = Eigen::VectorXf::Random(15);//Eigen::VectorXf::Zero(16);

    auto options = torch::TensorOptions().dtype(torch::kFloat);
    for (int i = 0; i < 1000; i++){

      input = Eigen::VectorXf::Random(15);
      std::vector<torch::jit::IValue> inputs_torch;
      torch::Tensor input_torch = torch::from_blob(input.data(), {15}, options).to(torch::kFloat);
      inputs_torch.push_back(input_torch);

      Eigen::VectorXf co1 = weight_control_net_layer_1 * input + bias_control_net_layer_1;
      Eigen::VectorXf ca1 = co1.array().tanh();
      Eigen::VectorXf co2 = weight_control_net_layer_2 * ca1 + bias_control_net_layer_2;
      Eigen::VectorXf ca2 = co2.array().tanh();
      Eigen::VectorXf co3 = weight_control_net_layer_3 * ca2 + bias_control_net_layer_3;
      Eigen::VectorXf ca3 = co3.array().tanh();
      Eigen::VectorXf input_allocation_net = ca3;

      Eigen::VectorXf scaled_thrust = input_allocation_net(Eigen::seq(0,2)).cwiseProduct(max_thrust - min_thrust)/2 + (max_thrust + min_thrust)/2;
      Eigen::VectorXf scaled_torque = input_allocation_net(Eigen::seq(3,5)).cwiseProduct(max_torque - min_torque)/2 + (max_torque + min_torque)/2;

      Eigen::VectorXf scaled_input_allocation_net = Eigen::VectorXf::Zero(6);
      scaled_input_allocation_net << scaled_thrust, scaled_torque;

      Eigen::VectorXf ao1 = weight_allocation_net_layer_1 * scaled_input_allocation_net + bias_allocation_net_layer_1;
      Eigen::VectorXf aa1 = ao1.cwiseMax(0);
      Eigen::VectorXf ao2 = weight_allocation_net_layer_2 * aa1 + bias_allocation_net_layer_2;

      Eigen::VectorXf output_allocation_net = ao2;

      torch::Tensor torch_tensor = converter.nn_model.forward(inputs_torch).toTensor();
      float* torch_data = torch_tensor.data_ptr<float>();
      Eigen::Map<Eigen::VectorXf> torch_vector(torch_data, torch_tensor.size(0));

      Eigen::VectorXf diff = torch_vector - output_allocation_net;
      double max_diff = diff.cwiseAbs().maxCoeff();
      if (max_diff > 2e-5){
        std::cout << "max_diff: " << max_diff << std::endl;
        std::cerr << "torch model and Eigen model not coherent";
      }
    }

    return 0;
}

